"""stage 스크립트 단위 테스트 (spec 6.1 의 6, 8)."""

import numpy as np
import torch

from conftest import load_stage
from kexp.config import CFG


# 6. eval chunking --------------------------------------------------------------

class _FakePredictor:
    """generate 호출마다 전역 난수를 배치 단위로 소비한다 (Kronos 의 샘플링과 같은 성질)."""

    def generate(self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose):
        return torch.rand(len(x), pred_len, 6).numpy() + x[:, -1:, :]


def test_eval_chunking_matches_unchunked_path():
    s5 = load_stage("stage5")
    n, T = 8, 4
    x = np.random.default_rng(0).standard_normal((n, T, 6)).astype(np.float32)
    stamp = np.zeros((n, T, 5), dtype=np.float32)
    run = lambda xs, seed, eb=None: s5.run_config(_FakePredictor(), None, xs, stamp[: len(xs)],   # noqa: E731
                                                  stamp[: len(xs)], CFG, None, None, None, None, seed, eb)
    whole = run(x, 0)
    assert np.array_equal(run(x, 0, n), whole)             # eval_batch = n_eval: 분할 없는 기존 경로
    assert np.array_equal(run(x, 0, 2 * n), whole)
    half = run(x, 0, n // 2)
    assert np.array_equal(half[: n // 2], run(x[: n // 2], 0))   # chunk 0 = 앞 절반만 seed 로 돌린 것
    assert np.array_equal(half[n // 2:], run(x[n // 2:], 1))     # chunk 1 의 seed 는 seed + 1
    assert not np.array_equal(half, whole)                 # 배치 구성이 바뀌면 값이 바뀐다


# 8. Stage 6 fast 판정 -----------------------------------------------------------

def _results(a_pos, j_pos, l_neg, n=64, lam=0.25, j_spread=0.02, ref=None):
    """양(음)의 기울기 비율을 지정한 합성 results.json."""
    rng = np.random.default_rng(0)

    def slopes(frac_pos, spread=0.02):
        k = int(round(frac_pos * n))
        mag = np.abs(rng.normal(0.0, spread, n)) + 1e-4
        return (np.where(np.arange(n) < k, 1.0, -1.0) * mag).tolist()

    base = rng.normal(0.0, 0.02, n).tolist()
    arms = {"A_paper_median_matrix_all": slopes(a_pos), "L_ctrl_reverse": slopes(1.0 - l_neg)}
    arms.update({f"J_ctrl_random_s{s}": slopes(j_pos, j_spread) for s in range(3)})
    res = {}
    for a, sl in arms.items():
        res[f"{a}|0.0"] = {"slopes": base}
        res[f"{a}|{lam}"] = {"slopes": sl}
    if ref is not None:
        res["REF_trend_unsteered|0.0"] = {"slopes": ref}
    return res, np.asarray(base), [0.0, lam]


def test_fast_verdict_paths():
    s6 = load_stage("stage6")

    res, base, lams = _results(a_pos=1.0, j_pos=0.5, l_neg=1.0)
    v = s6.fast_verdict(res, lams, base)
    assert v["A_up"] and v["lambda_star"] == 0.25
    assert v["verdict"] == "direction_specific" and v["reversible"] is True
    assert v["collapse_generic"] is False and v["ref_match"] is None

    res, base, lams = _results(a_pos=1.0, j_pos=1.0, l_neg=0.5, j_spread=0.0005)
    v = s6.fast_verdict(res, lams, base)
    assert v["verdict"] == "generic_perturbation" and v["reversible"] is False
    assert v["collapse_generic"] is True          # 랜덤 방향에서도 분산이 붕괴

    res, base, lams = _results(a_pos=0.5, j_pos=0.5, l_neg=0.5)
    v = s6.fast_verdict(res, lams, base)
    assert not v["A_up"] and v["lambda_star"] is None and v["verdict"] == "no_effect"

    res, base, lams = _results(a_pos=1.0, j_pos=0.8, l_neg=1.0)
    assert s6.fast_verdict(res, lams, base)["verdict"] == "ambiguous"


def test_fast_verdict_lambda_star_is_minimum_and_ref_match():
    s6 = load_stage("stage6")
    res, base, _ = _results(a_pos=0.6, j_pos=0.5, l_neg=1.0, lam=0.1)
    more, _, _ = _results(a_pos=1.0, j_pos=0.5, l_neg=1.0, lam=0.25)
    most, _, _ = _results(a_pos=1.0, j_pos=0.5, l_neg=1.0, lam=0.5)
    for extra in (more, most):
        res.update({k: v for k, v in extra.items() if not k.endswith("|0.0")})
    a_star = res["A_paper_median_matrix_all|0.25"]["slopes"]
    res["REF_trend_unsteered|0.0"] = {"slopes": list(a_star)}       # A 와 같은 분포
    v = s6.fast_verdict(res, [0.0, 0.1, 0.25, 0.5], base)
    assert v["lambda_star"] == 0.25                                  # 조건을 만족하는 최소 lambda
    assert v["ref_match"] is True
    res["REF_trend_unsteered|0.0"] = {"slopes": (-np.abs(a_star) - 1.0).tolist()}
    assert s6.fast_verdict(res, [0.0, 0.1, 0.25, 0.5], base)["ref_match"] is False


def test_fast_verdict_absent_without_fast_arms():
    s6 = load_stage("stage6")
    res = {"B_paper_mean_matrix_all|0.0": {"slopes": [0.1, -0.1]}}
    assert s6.fast_verdict(res, [0.0], np.array([0.1, -0.1])) is None
