"""kexp 패키지 단위 테스트 (spec 6.1 의 1, 2, 3, 4, 5, 7)."""

import numpy as np
import torch
from torch import nn

from kexp import intervene as IV
from kexp import kronos_loader as kl
from kexp import paths, runmeta
from kexp.hooks import ActivationRecorder


# 1. paths: 환경변수 설정 / 미설정 -----------------------------------------------

def test_paths_env(monkeypatch, tmp_path):
    monkeypatch.setenv("KEXP_OUT", str(tmp_path / "o"))
    monkeypatch.setenv("KEXP_SCRATCH", str(tmp_path / "s"))
    assert paths.out_root() == tmp_path / "o"
    assert paths.scratch_root() == tmp_path / "s"
    assert paths.results_dir("v2") == tmp_path / "o" / "results" / "v2"
    assert paths.activations_dir("v2", "ou/base") == tmp_path / "s" / "activations" / "v2" / "ou/base"


def test_paths_default(monkeypatch):
    monkeypatch.delenv("KEXP_OUT", raising=False)
    monkeypatch.delenv("KEXP_SCRATCH", raising=False)
    assert paths.out_root() == paths.REPO_ROOT / "experiment" / "_out"
    assert paths.scratch_root() == paths.REPO_ROOT / "experiment" / "_scratch"


# 2, 3. 랜덤 / 역방향 payload ---------------------------------------------------

def _S(L=3, T=16, D=832, seed=12345):
    # seed 0 은 쓰지 않는다. numpy 는 default_rng([0, 0]) 과 default_rng(0) 을 같은 난수열로
    # 만들기 때문에(뒤쪽 0 은 무시된다) payload 의 (seed 0, layer 0) 방향과 S 가 같아져 버린다.
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((L, T, D)) * rng.uniform(0.5, 3.0, (L, T, 1))).astype(np.float32)


def test_random_payload_matches_norm_and_is_seeded():
    S, lam = _S(), np.array([2.0, 1.0, 0.5])
    p0, cos = IV.build_random_payload(S, [0, 1, 2], lam, seed=0)
    for i in range(3):
        want = lam[i] * np.linalg.norm(S[i], axis=-1)
        got = p0[i].norm(dim=-1).numpy()
        assert np.max(np.abs(got - want) / want) < 1e-5          # 위치별 노름 = S 노름
        assert cos[i] < 0.1                                      # 사실상 직교
    again, _ = IV.build_random_payload(S, [0, 1, 2], lam, seed=0)
    other, _ = IV.build_random_payload(S, [0, 1, 2], lam, seed=1)
    assert all(torch.equal(p0[i], again[i]) for i in range(3))
    assert not torch.equal(p0[0], other[0])
    # 개입 레이어 집합이 달라도 같은 레이어에는 같은 방향이 나온다
    sub, _ = IV.build_random_payload(S, [2], lam, seed=0)
    assert torch.equal(sub[2], p0[2])


def test_reverse_payload_is_negated():
    S, lam = _S(D=64), np.array([2.0, 1.0, 0.5])
    a = IV.build_payload(None, S, [0, 2], lam, "matrix")
    r = IV.build_payload(None, S, [0, 2], -lam, "matrix")
    assert all(torch.equal(r[i], -a[i]) for i in (0, 2))


# 4. Steerer 의 AR 롤링 위치 정합 -------------------------------------------------

class _Block(nn.Module):
    def forward(self, x, key_padding_mask=None):
        return x


class _FakeModel(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.transformer = nn.ModuleList([_Block() for _ in range(n_layers)])

    def decode(self, x):
        for blk in self.transformer:
            x = blk(x)
        return x


def _steered_last_layer(model, payload, T, seq_lens):
    """디코딩 스텝마다 길이 seq_len 의 윈도우를 통과시키고 마지막 층 출력을 모은다.

    recorder 는 Steerer **안쪽에서** 등록한다. 먼저 등록하면 수정 전 출력을 잡는다.
    """
    D = next(iter(payload.values())).shape[-1]
    outs = []
    with IV.Steerer(model, list(payload), payload, T, form="matrix", scope="all"):
        with ActivationRecorder(model, dtype=None) as rec:
            for s in seq_lens:
                model.decode(torch.zeros(2, s, D))
            outs = [t[0] for t in rec.acts[len(model.transformer) - 1]]
    return outs


def test_steerer_rolling_alignment_with_random_payload():
    T, D = 6, 8
    payload, _ = IV.build_random_payload(_S(L=2, T=T, D=D), [0, 1], np.ones(2), seed=3)
    total = payload[0] + payload[1]
    model = _FakeModel(2)
    # max_context == T: 윈도우 길이는 T 로 고정되고 스텝마다 한 칸씩 밀린다.
    # 스텝 k 의 윈도우 위치 j 는 원래 인덱스 j + k - 1, 새로 생성된 토큰은 마지막 행.
    for k, out in enumerate(_steered_last_layer(model, payload, T, [T, T, T]), start=1):
        idx = torch.clamp(torch.arange(T) + k - 1, max=T - 1)
        assert torch.allclose(out, total[idx])
    # max_context > T: 윈도우가 한 칸씩 길어지고 위치는 밀리지 않는다.
    for k, out in enumerate(_steered_last_layer(model, payload, T, [T, T + 1, T + 2]), start=1):
        idx = torch.clamp(torch.arange(T + k - 1), max=T - 1)
        assert torch.allclose(out, total[idx])


# 5. randomize_model -----------------------------------------------------------

def _tiny_kronos():
    from model.kronos import Kronos
    torch.manual_seed(0)
    return Kronos(s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=4, ff_dim=64,
                  ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
                  token_dropout_p=0.0, learn_te=True)


def test_randomize_model():
    model, tok = _tiny_kronos(), nn.Linear(4, 4)        # tok: 토크나이저 자리의 아무 모듈
    before, tok_before = kl.param_hash(model), kl.param_hash(tok)
    kl.randomize_model(model, 0, tok)
    h0 = kl.param_hash(model)
    assert h0 != before and kl.param_hash(tok) == tok_before
    for m in model.modules():                          # reset_parameters 가 없는 모듈의 1차원 weight = 1
        if not hasattr(m, "reset_parameters"):
            for name, p in m.named_parameters(recurse=False):
                if p.dim() == 1 and name == "weight":
                    assert torch.all(p == 1.0)
    assert kl.param_hash(kl.randomize_model(_tiny_kronos(), 0)) == h0     # 같은 seed -> 같은 가중치
    assert kl.param_hash(kl.randomize_model(_tiny_kronos(), 1)) != h0


# 7. runmeta 체크섬 --------------------------------------------------------------

def test_checksum_roundtrip(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / ".done").mkdir()
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "results" / "b.bin").write_bytes(b"\x00" * 64)
    (tmp_path / "logs" / "x.log").write_text("log")
    (tmp_path / ".done" / "step").write_text("t")
    assert runmeta.checksum_write(tmp_path, "t") == 0
    assert runmeta.checksum_verify(tmp_path) == 0
    (tmp_path / "logs" / "x.log").write_text("log grows")           # 로그와 마커는 대상이 아니다
    assert runmeta.checksum_verify(tmp_path) == 0
    (tmp_path / "results" / "b.bin").write_bytes(b"\x00" * 63 + b"\x01")   # 1 바이트 변경
    assert runmeta.checksum_verify(tmp_path) == 1
    (tmp_path / "results" / "b.bin").write_bytes(b"\x00" * 64)
    assert runmeta.checksum_verify(tmp_path) == 0
    (tmp_path / "extra.txt").write_text("unlisted")
    assert runmeta.checksum_verify(tmp_path) == 1
