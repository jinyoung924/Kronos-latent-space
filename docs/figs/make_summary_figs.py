"""docs/final_report_요약.md 의 그림 7~13 을 저장된 산출물에서 다시 그린다.

실행 (저장소 루트에서):
    ~/.venvs/kronos/bin/python docs/figs/make_summary_figs.py [--run-id v2_fast]

그림 하나에는 패널 하나만 두고, 제목에는 그 그림이 말하려는 것을 적는다.
입력은 experiment/runs/<run_id> 의 data/ 와 results/ 뿐이다 (모델을 다시 돌리지 않는다).
사후 분석의 계산 정의는 docs/final_report.md 부록 C 를 따른다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# 색은 대상에 고정한다. 그림이 달라도 같은 대상은 같은 색이다.
C_CONST = "#898781"                              # constant 입력, 랜덤 방향 대조군 (중립 회색)
C_TREND = "#2a78d6"                              # trend 클래스
C_STEER = "#eb6834"                              # steering (+S)
C_REV = "#1baf7a"                                # 역방향 (−S)
C_STEER_LO, C_STEER_HI = "#f2a37f", "#c94d16"    # steering 세기 2단계 (PCA 그림)
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

LAYER, T_LAST, CH_CLOSE = 11, 511, 3
LAMS = [0.0, 0.1, 0.15, 0.25, 0.5]


def setup_style() -> None:
    plt.rcParams.update({
        # Apple SD Gothic Neo 에는 U+2212(−)가 없다. 없는 글리프는 뒤의 글꼴에서 가져온다
        "font.family": ["Apple SD Gothic Neo", "Arial", "NanumGothic", "DejaVu Sans"],
        "font.size": 11,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK2, "axes.labelsize": 11, "axes.labelpad": 8,
        "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False, "legend.fontsize": 10.5,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def new_fig(title: str, subtitle: str, legend: bool = False, right: float = 0.96):
    """제목(주장) + 부제(무엇을 그렸나) 아래에 패널 하나. legend=True 면 범례 한 줄 자리를 남긴다."""
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    fig.subplots_adjust(left=0.105, right=right, top=0.775 if legend else 0.83, bottom=0.135)
    fig.text(0.022, 0.962, title, fontsize=15, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.022, 0.892, subtitle, fontsize=10.5, color=INK2, ha="left", va="top")
    return fig, ax


def legend_row(ax, **kw) -> None:
    leg = ax.legend(loc="lower left", bbox_to_anchor=(-0.012, 1.015), borderaxespad=0,
                    ncol=kw.pop("ncol", 4), handlelength=1.5, handletextpad=0.5,
                    columnspacing=1.6, **kw)
    for h in leg.legend_handles:
        h.set_alpha(1.0)


def save(fig, out: Path) -> None:
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  {out.relative_to(ROOT)}")


def signed(v: float, nd: int = 2) -> str:
    return f"{v:+.{nd}f}".replace("-", "−")


def pct(v: float) -> str:
    return f"{v:.1f}".removesuffix(".0") + "%"


def znorm_close(x: np.ndarray) -> np.ndarray:
    """Kronos 전처리와 같은 창별·채널별 z-score(ddof=0, ε=1e-5, ±5 로 자름) 뒤의 종가. [N, T]"""
    mean, std = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True)
    return np.clip((x - mean) / (std + 1e-5), -5, 5)[:, :, CH_CLOSE]


def unit(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float64)
    return v / np.linalg.norm(v)


def last_token_acts(res_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """layer 11, 마지막 위치의 활성화. (constant, trend), 각 [앞 512표본, 832]"""
    sub = np.load(res_dir / "pca_subset.npz")
    assert int(sub["tokens"][0]) == T_LAST
    act = sub["act"][:, LAYER, 0].astype(np.float64)
    return act[0], act[1]


# ── 그림 7, 8: steering 세기에 따른 상승 예측 비율 ──────────────────────────────

def fig_up_rate(run: Path, noise: str, title: str, out: Path) -> None:
    res = json.loads((run / f"results/v2/{noise}_base/steer/results.json").read_text())

    def rate(arm: str) -> np.ndarray:
        return np.array([100 * res[f"{arm}|{l}"]["frac_positive"] for l in LAMS])

    a, rev = rate("A_paper_median_matrix_all"), rate("L_ctrl_reverse")
    j = np.stack([rate(f"J_ctrl_random_s{s}") for s in range(3)])
    n = len(res["A_paper_median_matrix_all|0.0"]["slopes"])

    fig, ax = new_fig(
        title,
        f"constant 입력 {n}개 가운데 예측 종가의 기울기가 양수인 비율 · Kronos-base · "
        "랜덤 방향은 seed 3개 평균 (띠: seed 최소~최대)",
        legend=True, right=0.845)

    for y, txt in ((90, "steering 판정 기준: 상승 90% 이상"),
                   (10, "역방향 판정 기준: 하락 90% 이상 (= 상승 10% 이하)")):
        ax.axhline(y, color=INK2, lw=1.0, ls=(0, (5, 3)), zorder=1)
        ax.text(0.008, y + 1.6, txt, fontsize=9.5, color=INK2, va="bottom")

    ax.fill_between(LAMS, j.min(axis=0), j.max(axis=0), color=C_CONST, alpha=0.16, lw=0, zorder=2)
    mk = dict(marker="o", ms=8, mec="white", mew=1.6)
    ax.plot(LAMS, j.mean(axis=0), color=C_CONST, lw=2, zorder=3, label="랜덤 방향 (S와 같은 크기)", **mk)
    ax.plot(LAMS, rev, color=C_REV, lw=2, zorder=4, label="역방향 (−λ·S)", **mk)
    ax.plot(LAMS, a, color=C_STEER, lw=2.4, zorder=5, label="steering (+λ·S)", **mk)
    ax.plot([0], [a[0]], "o", ms=8, color=INK2, mec="white", mew=1.6, zorder=6)

    # 오른쪽 끝 값. 서로 겹치면 위아래로 조금 벌린다.
    ends = sorted([[a[-1], pct(a[-1]), True],
                   [j.mean(axis=0)[-1], pct(j.mean(axis=0)[-1]), False],
                   [rev[-1], f"{pct(rev[-1])} (하락 {pct(100 - rev[-1])})", False]])
    ypos = [e[0] for e in ends]
    for i in range(1, len(ypos)):
        if ypos[i] - ypos[i - 1] < 6.5:
            mid = (ypos[i] + ypos[i - 1]) / 2
            ypos[i - 1], ypos[i] = mid - 3.25, mid + 3.25
    for (_, txt, bold), y in zip(ends, ypos):
        ax.annotate(txt, (LAMS[-1], y), xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=10.5, color=INK, fontweight="bold" if bold else "normal",
                    annotation_clip=False)

    ax.set_xlim(-0.025, 0.52)
    ax.set_ylim(-4, 106)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(LAMS, [f"0\n(개입 없음: {pct(a[0])})", "0.1", "0.15", "0.25", "0.5"])
    ax.set_xlabel("steering 세기 λ_rel (더한 벡터의 크기 ÷ 원래 활성화의 크기)")
    ax.set_ylabel("상승으로 예측한 비율 (%)")
    handles, labels = ax.get_legend_handles_labels()
    legend_row(ax, handles=handles[::-1], labels=labels[::-1])
    save(fig, out)


# ── 수준 되돌림: 입력의 마지막 가격 수준 대 예측 기울기, 개입 전후 ──────────────

def fig_level_reversion(run: Path, out: Path, lam: float = 0.25) -> None:
    """가로: constant 입력의 마지막 정규화 종가 z_T, 세로: 64스텝 예측 종가의 기울기.
    개입 없음과 steering(λ_rel=lam)을 같은 입력 64개에 대해 겹쳐 그린다."""
    res_dir = run / "results/v2/ou_base"
    res = json.loads((res_dir / "steer/results.json").read_text())
    base = np.array(res["A_paper_median_matrix_all|0.0"]["slopes"])
    steer = np.array(res[f"A_paper_median_matrix_all|{lam}"]["slopes"])
    # Stage 5 의 평가 입력 = held-out 인덱스의 앞 n개 (base 클래스)
    idx = np.load(res_dir / "ldr.npz")["idx_test"][:len(base)]
    z_last = znorm_close(np.load(run / "data/v2/ou/base.npz")["x"][idx])[:, T_LAST]

    fits = {}
    for name, y in (("base", base), ("steer", steer)):
        b1, b0 = np.polyfit(z_last, y, 1)
        fits[name] = (b0, b1, np.corrcoef(z_last, y)[0, 1])

    fig, ax = new_fig(
        "개입 전 예측은 '어디서 끝났나'를 따라가고, steering은 그 반응을 지운다",
        f"점 하나 = constant 입력 하나 ({len(base)}개, 같은 입력을 두 번) · OU · 세로는 64스텝 예측 종가에 맞춘 "
        "직선의 기울기 · 선은 회귀직선",
        legend=True)
    ax.grid(True, axis="x")
    ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
    xs = np.linspace(z_last.min() - 0.2, z_last.max() + 0.2, 2)
    for name, y, c, label in (("base", base, C_CONST, "개입 없음"),
                              ("steer", steer, C_STEER, f"steering (+λ·S, λ_rel {lam})")):
        b0, b1, r = fits[name]
        ax.scatter(z_last, y, s=26, color=c, alpha=0.65, lw=0, zorder=3, label=label)
        ax.plot(xs, b0 + b1 * xs, color=c, lw=1.6, zorder=2)
    b0, b1, r = fits["base"]
    ax.text(0.985, 0.955, f"개입 없음: 높게 끝날수록 하락 예측 (r = {signed(r)}, 기울기 {signed(b1, 4)}/z)",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5, color=INK,
            bbox=dict(facecolor="white", edgecolor="none", pad=2))
    b0, b1, r = fits["steer"]
    # 왼쪽 아래는 점이 없는 자리다. 거기서 steering 회귀직선으로 화살표를 긋는다
    x_arrow = np.percentile(z_last, 12)
    ax.annotate(f"steering: 어디서 끝났든 {signed(steer.min(), 4)} ~ {signed(steer.max(), 4)}\n"
                f"(모두 양수, 입력에 대한 반응은 {signed(b1, 4)}/z로 거의 사라짐)",
                xy=(x_arrow, b0 + b1 * x_arrow), xycoords="data",
                xytext=(0.02, 0.07), textcoords="axes fraction",
                ha="left", va="bottom", fontsize=10.5, color=C_STEER_HI, zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=C_STEER_HI, lw=1.4, shrinkA=2, shrinkB=4,
                                mutation_scale=12))
    ax.set_xlabel("입력의 마지막 정규화 종가 z_T (0 = 창 평균, 양수 = 평균보다 높게 끝남)")
    ax.set_ylabel("예측 종가의 기울기 (정규화 단위 / 스텝)")
    ax.set_xticks([-2, -1, 0, 1, 2], ["−2", "−1", "0", "+1", "+2"])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.2f}".replace("-", "−") if v else "0")
    ax.margins(x=0.08, y=0.12)
    legend_row(ax)
    save(fig, out)


# ── 그림 9: 정규화 뒤 위치별 가격 수준 ─────────────────────────────────────────

def level_profiles(run: Path) -> tuple[np.ndarray, np.ndarray]:
    zb = znorm_close(np.load(run / "data/v2/ou/base.npz")["x"])
    zt = znorm_close(np.load(run / "data/v2/ou/trend.npz")["x"])
    return zb, zt


def fig_input_level(run: Path, out: Path) -> None:
    zb, zt = level_profiles(run)
    t = np.arange(zb.shape[1])
    fig, ax = new_fig(
        "정규화하면 상승 추세는 '앞은 낮고 끝은 높은 가격 수준'이 된다",
        f"OU 입력 {len(zb)}쌍 · 창별 z-score 정규화 뒤 종가의 위치별 중앙값 (띠: 25~75%) · "
        "숫자는 trend − constant",
        legend=True)
    med = {}
    for z, c, name in ((zb, C_CONST, "constant (추세 없음)"), (zt, C_TREND, "trend (상승 추세)")):
        q25, q50, q75 = np.percentile(z, [25, 50, 75], axis=0)
        ax.fill_between(t, q25, q75, color=c, alpha=0.14, lw=0)
        ax.plot(t, q50, color=c, lw=2, label=name)
        med[c] = q50
    d = med[C_TREND] - med[C_CONST]
    mid = len(t) // 2
    for ti, ha, dx, dy, name in ((0, "left", 14, -16, "첫 봉"), (mid, "center", 0, -34, "가운데"),
                                 (T_LAST, "right", -14, 14, "마지막 봉")):
        txt = "0" if abs(d[ti]) < 0.005 else signed(d[ti])
        ax.plot([ti], [med[C_TREND][ti]], "o", ms=8, color=C_TREND, mec="white", mew=1.6, zorder=5,
                clip_on=False)
        ax.annotate(f"{name}: {txt}", (ti, med[C_TREND][ti]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=11, color=INK,
                    fontweight="bold")
    ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
    ax.set_xlim(-6, T_LAST + 6)
    ax.set_ylim(-2.2, 2.2)
    ax.set_xticks([0, 128, 256, 384, T_LAST])
    ax.set_yticks([-2, -1, 0, 1, 2], ["−2", "−1", "0", "+1", "+2"])
    ax.set_xlabel("입력 창 안의 위치 t (0 = 첫 봉, 511 = 마지막 봉)")
    ax.set_ylabel("정규화된 종가 (창 평균 = 0, 표준편차 = 1)")
    legend_row(ax)
    save(fig, out)


# ── 그림 10: 위치별 steering 벡터의 방향 ───────────────────────────────────────

def fig_s_direction(run: Path, out: Path) -> None:
    S = np.load(run / "results/v2/ou_base/steering.npz")["S_matrix_median"][LAYER].astype(np.float64)
    norm = np.linalg.norm(S, axis=1)
    cos = S @ S[T_LAST] / (norm * norm[T_LAST])
    zb, zt = level_profiles(run)
    r = np.corrcoef(cos, np.median(zt, axis=0) - np.median(zb, axis=0))[0, 1]
    cross = int(np.where(cos < 0)[0].max())              # 이 위치 다음부터 끝까지 양수
    t = np.arange(len(cos))

    fig, ax = new_fig(
        "steering 벡터 S도 창 가운데에서 방향이 뒤집힌다",
        f"위치별 S(t)가 마지막 위치의 S와 이루는 코사인 · OU, layer {LAYER} · "
        f"입력의 수준 차이(trend − constant)와 위치별 상관 {r:.2f}")
    ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
    ax.plot(t, cos, color=C_STEER, lw=2, zorder=3)
    ax.plot([cross + 0.5, T_LAST], [0, 1], "o", ms=8, color=C_STEER, mec="white", mew=1.6, zorder=5,
            clip_on=False)
    front = cos[8:225]
    ax.text(150, front.min() - 0.11,
            f"앞쪽 위치: 마지막 위치와 반대 방향 (코사인 {signed(front.max())} ~ {signed(front.min())})",
            ha="center", va="top", fontsize=10.5, color=INK)
    ax.annotate(f"t ≈ {cross}에서 0을 지남", (cross + 0.5, 0), xytext=(12, -16),
                textcoords="offset points", ha="left", va="center", fontsize=10.5, color=INK)
    ax.annotate("마지막 위치 (기준이므로 +1)", (T_LAST, 1), xytext=(-2, 17), textcoords="offset points",
                ha="right", va="center", fontsize=10.5, color=INK)
    ax.set_xlim(-6, T_LAST + 6)
    ax.set_ylim(-1.08, 1.27)
    ax.set_xticks([0, 128, 256, 384, T_LAST])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1], ["−1", "−0.5", "0", "+0.5", "+1"])
    ax.set_xlabel("입력 창 안의 위치 t (0 = 첫 봉, 511 = 마지막 봉)")
    ax.set_ylabel("마지막 위치의 S와의 코사인 (+1 같은 방향, −1 반대 방향)")
    save(fig, out)


# ── 그림 11: S 방향 성분 대 입력의 마지막 가격 수준 ────────────────────────────

def fig_s_vs_level(run: Path, out: Path) -> None:
    res_dir = run / "results/v2/ou_base"
    a, _ = last_token_acts(res_dir)
    s = unit(np.load(res_dir / "steering.npz")["S_matrix_median"][LAYER][T_LAST])
    z_last = level_profiles(run)[0][:len(a), T_LAST]
    proj = a @ s
    r = np.corrcoef(z_last, proj)[0, 1]

    fig, ax = new_fig(
        f"S 방향 성분은 입력의 '마지막 가격 수준'을 따라간다 (r = {signed(r)})",
        f"점 하나 = constant 입력 하나 ({len(a)}개) · OU, layer {LAYER}, 마지막 위치의 활성화를 S 방향에 투영 · "
        "r은 피어슨 상관")
    ax.grid(True, axis="x")
    ax.scatter(z_last, proj, s=20, color=C_CONST, alpha=0.6, lw=0, zorder=3)
    ax.set_xlabel("입력의 마지막 정규화 종가 z_T (0 = 창 평균, 양수 = 평균보다 높게 끝남)")
    ax.set_ylabel("활성화의 S 방향 성분 (투영값)")
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3], ["−3", "−2", "−1", "0", "+1", "+2", "+3"])
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1000))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}".replace("-", "−"))
    ax.margins(x=0.07, y=0.08)
    save(fig, out)


# ── 그림 12: 프로브 방향 대 S 방향 ─────────────────────────────────────────────

def fig_probe_vs_s(run: Path, out: Path) -> None:
    res_dir = run / "results/v2/ou_base"
    a, b = last_token_acts(res_dir)
    ldr = np.load(res_dir / "ldr.npz")
    w = unit(ldr["w_last"][LAYER])
    s = unit(np.load(res_dir / "steering.npz")["S_matrix_median"][LAYER][T_LAST])
    te = np.sort(ldr["idx_test"][ldr["idx_test"] < len(a)])     # 프로브 학습에 쓰지 않은 표본만

    fig, ax = new_fig(
        "두 클래스는 프로브 방향으로 갈리고, S 방향으로는 겹친다",
        f"점 하나 = 평가용 입력 하나 (클래스당 {len(te)}개) · OU, layer {LAYER}, 마지막 위치의 활성화 · "
        f"두 방향의 코사인 {w @ s:.2f}",
        legend=True)
    ax.grid(True, axis="x")
    for X, c, name in ((a[te], C_CONST, "constant (추세 없음)"), (b[te], C_TREND, "trend (상승 추세)")):
        ax.scatter(X @ s, X @ w, s=20, color=c, alpha=0.6, lw=0, zorder=3, label=name)
    ax.margins(x=0.08, y=0.08)
    ax.text(0.015, 0.97, "눈금 주의: 가로 1칸은 1,000, 세로 1칸은 10 (100배 차이)",
            transform=ax.transAxes, ha="left", va="top", fontsize=10, color=INK2, zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=3))
    ax.set_xlabel("steering 벡터 S 방향 성분 (예측을 움직인 방향)")
    ax.set_ylabel("프로브 방향 성분 (두 클래스를 가르는 방향)")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1000))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}".replace("-", "−"))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}".replace("-", "−"))
    legend_row(ax, markerscale=1.6)
    save(fig, out)


# ── 그림 13: steering 뒤 표현의 PCA ────────────────────────────────────────────

def fig_pca_steered(run: Path, out: Path) -> None:
    res_dir = run / "results/v2/ou_base"
    a, b = last_token_acts(res_dir)
    st = np.load(res_dir / "steering.npz")
    # Stage 6 과 같은 정의: 끝 구간의 S 를 λ_rel·‖h‖/‖S‖ 배 해서 constant 활성화에 더한다
    S = st["S_median_late"][LAYER].astype(np.float64)
    lam_abs = {l: l * st["h_norm_l2"][LAYER] / np.linalg.norm(S) for l in (0.25, 0.5)}

    X = np.vstack([a, b])
    mu = X.mean(axis=0)
    _, sv, vt = np.linalg.svd(X - mu, full_matrices=False)
    pcs, ev = vt[:2].copy(), sv[:2] ** 2 / np.sum(sv ** 2)
    if (b.mean(axis=0) - a.mean(axis=0)) @ pcs[0] < 0:          # trend 가 오른쪽에 오도록
        pcs[0] *= -1
    if np.median((X - mu) @ pcs[1]) < 0:
        pcs[1] *= -1

    fig, ax = new_fig(
        "steering된 표현은 옆으로 밀려날 뿐, trend 자리로 가지 않는다",
        f"OU, layer {LAYER}, 마지막 위치 활성화의 PCA 평면 (클래스당 {len(a)}개) · "
        "steering = constant 활성화 + λ·S",
        legend=True)
    ax.grid(True, axis="x")
    groups = [(a, C_CONST, "constant (개입 전)", 2),
              (a + lam_abs[0.25] * S, C_STEER_LO, "steering λ_rel 0.25", 3),
              (a + lam_abs[0.5] * S, C_STEER_HI, "steering λ_rel 0.5", 4),
              (b, C_TREND, "trend (실제 추세 입력)", 5)]
    proj = {}
    for Xg, c, name, z in groups:
        p = proj[c] = (Xg - mu) @ pcs.T
        ax.scatter(p[:, 0], p[:, 1], s=9, color=c, alpha=0.55, lw=0, zorder=z, label=name)

    # 호의 왼쪽 끝이 얼마나 옮겨졌는지 화살표로 보인다
    lo0, lo1 = (proj[c][proj[c][:, 1].argmin()] for c in (C_CONST, C_STEER_HI))
    ya = min(lo0[1], lo1[1]) - 200
    ax.annotate("", xy=(lo1[0], ya), xytext=(lo0[0], ya), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=C_STEER_HI, lw=1.8, mutation_scale=14,
                                shrinkA=0, shrinkB=0))
    ax.text((lo0[0] + lo1[0]) / 2, ya - 100, "steering: 호 전체가 오른쪽(PC1)으로 밀린다",
            ha="center", va="top", fontsize=10.5, color=INK, zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=2))
    leg_pts = proj[C_TREND][proj[C_TREND][:, 1] < np.percentile(proj[C_TREND][:, 1], 12)]
    ax.annotate("trend 표본이 모인 자리", np.median(leg_pts, axis=0), xytext=(-20, 0),
                textcoords="offset points", ha="right", va="center", fontsize=10.5, color=INK,
                zorder=6)
    ax.set_ylim(ya - 430, None)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"제1주성분 PC1 (분산의 {ev[0]:.0%})")
    ax.set_ylabel(f"제2주성분 PC2 (분산의 {ev[1]:.0%})")
    fmt = lambda v, _: f"{v:,.0f}".replace("-", "−")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 3, 1, 2]
    legend_row(ax, handles=[handles[i] for i in order], labels=[labels[i] for i in order],
               markerscale=2.2)
    save(fig, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default="v2_fast")
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()
    run = ROOT / "experiment/runs" / args.run_id
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()
    print(f"[make_summary_figs] {run.relative_to(ROOT)} -> {args.out.relative_to(ROOT)}")
    fig_up_rate(run, "ou", "OU: steering은 λ_rel 0.25부터 예측을 모두 상승으로 바꿨다",
                args.out / "steer_up_rate_ou.png")
    fig_up_rate(run, "rw", "RW: steering의 상승 비율은 85% 안팎에서 멈췄다 (기준 90% 미달)",
                args.out / "steer_up_rate_rw.png")
    fig_level_reversion(run, args.out / "level_reversion_ou.png")
    fig_input_level(run, args.out / "input_level_ou.png")
    fig_s_direction(run, args.out / "s_direction_ou.png")
    fig_s_vs_level(run, args.out / "s_vs_last_level_ou.png")
    fig_probe_vs_s(run, args.out / "probe_vs_s_ou.png")
    fig_pca_steered(run, args.out / "pca_steered_ou.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
