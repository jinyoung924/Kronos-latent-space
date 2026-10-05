"""Kronos 모델/토크나이저 로드 + 스펙 실측 + 전처리 재현.

전처리 함수를 여기에 모아두는 이유:
  Stage 2(활성화 추출)와 Stage 5(steering 추론)가 반드시 **동일한** 전처리를
  거쳐야 한다. KronosPredictor.predict() 안에 묻혀 있는 z-score/clip 로직을
  각 stage 가 따로 복붙하면 미묘하게 어긋난다.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
import torch

from kexp import paths

paths.bootstrap_sys_path()

from model.kronos import Kronos, KronosTokenizer, calc_time_stamps  # noqa: E402

PRICE_COLS = ["open", "high", "low", "close"]
FEAT_COLS = PRICE_COLS + ["volume", "amount"]
TIME_COLS = ["minute", "hour", "weekday", "day", "month"]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_kronos(model_cfg, device: torch.device | None = None):
    """토크나이저와 모델을 eval 모드로 로드한다.

    eval() 은 필수다. TransformerBlock 에 dropout 이 있어서 train 모드로 두면
    같은 입력에 대해 활성화가 매번 달라지고, LDR 이 노이즈로 오염된다.
    """
    if model_cfg.model_id.endswith("Kronos-base") and not (
            model_cfg.revision and model_cfg.tokenizer_revision):
        raise ValueError("Kronos-base 는 revision / tokenizer_revision 을 고정해야 한다 (kexp/config.py).")
    device = device or pick_device()
    tokenizer = KronosTokenizer.from_pretrained(model_cfg.tokenizer_id,
                                                revision=model_cfg.tokenizer_revision)
    model = Kronos.from_pretrained(model_cfg.model_id, revision=model_cfg.revision)
    tokenizer = tokenizer.to(device).eval()
    model = model.to(device).eval()
    return tokenizer, model, device


def set_determinism() -> None:
    """TF32 를 끈다. 모든 stage 가 시작할 때 호출한다.

    TF32 행렬곱은 빠르지만 결과를 바꾼다 (Kronos-investing 에서 실측).
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def param_hash(module) -> str:
    """파라미터 값의 해시. 가중치가 바뀌었는지 확인하는 용도."""
    h = hashlib.sha256()
    for name, p in module.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def randomize_model(model, seed: int, tokenizer=None, init: str = "kronos"):
    """무작위 초기화 대조군: 같은 구조, 학습되지 않은 가중치. 토크나이저는 건드리지 않는다.

    먼저 reset_parameters() 가 있는 하위 모듈은 그것을 호출하고, 없는 모듈이 직접 가진
    파라미터는 1차원 weight = 1, bias = 0, 2차원 이상 = N(0, 0.02^2) 로 채운다.
    난수열이 장치에 따라 달라지지 않도록 CPU 에서 초기화한 뒤 원래 장치로 돌려보낸다.

    init
        "torch"  : 위에서 끝낸다 (PyTorch 기본 초기화). 임베딩이 N(0, 1) 이라 블록 출력보다
                   훨씬 커서, base 구조에서 블록이 residual 에 층당 약 1% 만 기여한다. 대조군이
                   "학습 안 된 트랜스포머" 보다 "위치별 토큰 임베딩" 에 가깝다. 참고용이다.
        "kronos" : 이어서 Kronos 가 학습을 시작할 때 쓰는 초기화(model._init_weights:
                   Linear 는 xavier normal, Embedding 은 std = d_model^-0.5)를 적용한다.
                   블록 기여가 층당 30~50% 로 사전학습 모델과 비슷하다. 판정 기준이다.
    """
    if init not in ("kronos", "torch"):
        raise ValueError(f"알 수 없는 init: {init}")
    device = next(model.parameters()).device
    before = param_hash(model)
    tok_before = param_hash(tokenizer) if tokenizer is not None else None
    model.cpu()
    torch.manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
                continue
            for name, p in m.named_parameters(recurse=False):
                if p.dim() >= 2:
                    p.normal_(0.0, 0.02)
                elif name.endswith("bias"):
                    p.zero_()
                else:
                    p.fill_(1.0)
        if init == "kronos":
            model.apply(model._init_weights)
    model.to(device).eval()
    after = param_hash(model)
    print(f"  randomize_model(seed={seed}, init={init}): 모델 해시 {before} -> {after} "
          f"({'바뀜' if before != after else '안 바뀜 — 오류'})")
    if tokenizer is not None:
        tok_after = param_hash(tokenizer)
        print(f"  토크나이저 해시 {tok_before} -> {tok_after} "
              f"({'그대로' if tok_before == tok_after else '바뀜 — 오류'})")
        if tok_before != tok_after:
            raise RuntimeError("randomize_model 이 토크나이저를 바꿨다.")
    if before == after:
        raise RuntimeError("randomize_model 뒤에도 모델 파라미터가 그대로다.")
    return model


def describe(tokenizer, model) -> dict[str, Any]:
    """Stage 0 에서 확인해야 하는 실측 스펙."""
    return {
        "n_layers": len(model.transformer),
        "d_model": int(model.d_model),
        "n_heads": int(model.n_heads),
        "ff_dim": int(model.ff_dim),
        "s1_bits": int(model.s1_bits),
        "s2_bits": int(model.s2_bits),
        "s1_vocab_size": int(model.s1_vocab_size),
        "model_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "tokenizer_params_M": round(sum(p.numel() for p in tokenizer.parameters()) / 1e6, 2),
        "tokenizer_d_in": int(tokenizer.d_in),
        "tokenizer_d_model": int(tokenizer.d_model),
    }


# --- 전처리 (KronosPredictor.predict 와 동일하게 재현) -------------------------

def normalize_batch(x: np.ndarray, clip: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """채널별 z-score 후 [-clip, clip] 으로 자른다.

    Args:
        x: [B, T, 6] 원시 OHLCVA
    Returns:
        x_norm [B, T, 6], mean [B, 6], std [B, 6]

    Volume/Amount 가 전부 0 이면 mean=std=0 이라 (0-0)/(0+1e-5)=0 이 되어
    NaN 없이 0 이 유지된다 (명세 2절이 코드상 안전한 이유).
    """
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    x_norm = (x - mean[:, None, :]) / (std[:, None, :] + 1e-5)
    return np.clip(x_norm, -clip, clip), mean, std


def clip_fraction(x_norm_unclipped: np.ndarray, clip: float) -> float:
    """클리핑에 걸린 값의 비율 — Stage 1 캘리브레이션 지표 (명세 3절)."""
    return float((np.abs(x_norm_unclipped) > clip).mean())


def make_stamps(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """[T, 5] 시간 특징. 두 클래스가 반드시 동일한 값을 써야 한다."""
    return calc_time_stamps(pd.Series(timestamps)).values.astype(np.float32)


def to_tokens(tokenizer, x_norm: np.ndarray, device: torch.device):
    """[B, T, 6] -> (s1_ids, s2_ids) 각 [B, T].

    트랜스포머의 시퀀스 위치 수 = K-line 스텝 수 T 이다.
    HierarchicalEmbedding 이 (s1, s2) 두 토큰을 한 위치로 융합하기 때문.
    """
    xt = torch.from_numpy(np.ascontiguousarray(x_norm, dtype=np.float32)).to(device)
    with torch.no_grad():
        return tokenizer.encode(xt, half=True)
