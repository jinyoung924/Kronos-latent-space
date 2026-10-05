"""Stage 7 — fast track 요약 (spec 3.12).

끝난 노이즈의 요약 JSON 만 읽어 다섯 질문(Q0~Q4)에 답하는 한 장짜리 문서를 만든다.
모델도 활성화도 필요 없다. run_plan.py 가 노이즈 하나가 끝날 때마다 실행한다.

    입력  results/<tag>/<noise>_<model>/{stage3_summary, stage3_controls, stage4_summary,
          stage6_summary}.json, data/<tag>/<noise>/meta.json, experiment/expected/v1_baseline.json
    출력  <OUT>/fast_summary.md, <OUT>/fast_summary.json

해석 문장은 outline 8절 표의 문장 중에서 **선택**만 한다. 새 문장을 만들지 않는다.
개입 쪽(8.2)의 선택 규칙은 spec 2.2 의 사전 고정 판정이다. 표현 쪽(8.1)은 spec 에 수치
규칙이 없어 아래 Q1_RATIO 하나로 정했다 — 결론을 쓰기 전에 표의 비율을 직접 볼 것.

실행:
    python experiment/code/stages/stage7_fast_summary.py --model base --noise ou,rw
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kexp import paths  # noqa: E402
from kexp.config import CFG  # noqa: E402

# outline 6절의 재현 판정 기준 (제안값)
LDR_REL_TOL = 0.05      # 층별 held-out LDR: 상대오차 수 % 이내
COS_ABS_TOL = 0.01      # 코사인: 소수 둘째 자리까지
REF_COUNT_TOL = 2       # REF 양의 기울기 수: +-2/32
# outline 8.1 의 "≈ / 비슷" 과 "> / 증폭" 을 가르는 비율. spec 에 없는 값이다.
Q1_RATIO = 2.0

S81 = {   # outline 8.1
    "like_random": "분리는 입력·구조에서 오며, 학습된 표현이라는 근거 없음",
    "preserved": "입력 특성이 보존·전달됨. 개념 \"형성\"의 근거는 약함",
    "amplified": "모델이 문맥 누적으로 추세 구분을 만들어냄",
}
S82 = {   # outline 8.2
    "generic_perturbation": "효과는 섭동 일반의 성질. 개념 특이적 조정이 아님",
    "specific_unlike_ref": "방향 특이적 조정은 가능하나, 모델이 실제 추세 입력에 하는 계산과 "
                           "다른 출력을 주입 (1차 OU 잠정 결론)",
    "specific_like_ref": "개념 방향이 모델의 자연 기작과 일치",
    "no_effect": "해당 설정에서 인과적 사용의 증거 없음",
}
NO_ROW = "outline 8절 표에 해당하는 행 없음 — 수치 그대로 보고"


def load(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def f(v, spec=".3f") -> str:
    return "-" if v is None else format(v, spec)


def mark(ok) -> str:
    return {True: "일치", False: "불일치", None: "참고"}[ok]


def q0_rows(v1: dict, dmeta: dict, s3: dict, s4: dict, fast: dict | None, expected_fp) -> list:
    """Q0 재현표. Stage 5 지표는 subset32(앞 32 입력 = v1 평가 표본)로 비교한다."""
    rows = []

    def add(item, ref, val, ok, crit):
        rows.append({"item": item, "v1": ref, "v2": val, "ok": ok, "criterion": crit})

    add("data_fingerprint", expected_fp, dmeta["data_fingerprint"],
        None if expected_fp is None else dmeta["data_fingerprint"] == expected_fp, "완전 일치")
    v = dmeta["input_ldr_close_slope"]
    add("입력단 LDR (close 기울기)", v1["input_ldr"], round(v, 3), abs(v - v1["input_ldr"]) < 0.005,
        "소수 둘째 자리까지")
    L = s3["L"]
    b = s3["bands"]
    for key, label, val in (("layer0_mid", "held-out LDR layer 0, mid", b["mid"][0]),
                            ("layer1_mid", "held-out LDR layer 1, mid", b["mid"][1]),
                            ("layer11_late", f"held-out LDR layer {L-1}, late", b["late"][L - 1]),
                            ("layer11_last", f"held-out LDR layer {L-1}, last", b["last"][L - 1])):
        ref = v1["ldr_heldout"][key]
        add(label, ref, round(val, 2), abs(val - ref) / ref <= LDR_REL_TOL,
            f"상대오차 {LDR_REL_TOL:.0%} 이내")
    ratio = max(max(b["late"]) / (s3["null_test_max"] + 1e-12), 1e-12)
    add("null 대비 배율", v1["null_ratio"], round(ratio, 0),
        abs(math.log10(ratio / v1["null_ratio"])) < 0.5, "같은 자릿수")
    peaks = sorted(s3["smoothed_peak_token_by_layer"])
    med = peaks[len(peaks) // 2]
    lo, hi = v1["peak_t_range"]
    add("평활 LDR 정점 위치 (레이어 중앙값)", f"{lo}~{hi}", med, lo <= med <= hi, "1차 범위 안")
    for key, name, label in (("cos_mean_median", "mean/late vs median/late", "cos(mean, median)"),
                             ("cos_lda_vs_median", "lda(last) vs median/late", "cos(LDA, median)")):
        vals = sorted(s4["cosine"][name])
        val = vals[len(vals) // 2]
        add(label, v1[key], round(val, 4), abs(val - v1[key]) <= COS_ABS_TOL,
            f"차이 {COS_ABS_TOL} 이내")
    if fast:
        a = v1["steer_A"]
        t = {r["lambda"]: r["A_subset32"] for r in fast["table"]}
        at = t.get(a["lambda_up_min"])
        if at:
            add(f"A 팔 lambda={a['lambda_up_min']} 양의 기울기 (subset32)",
                f"{a['frac_positive_at_up']:.0%}", f"{at['n_positive']}/{at['n']}",
                at["frac_positive"] > 0.5 and at["p_sign"] < 0.01, "상향 분류·유의성(p<0.01) 동일")
        ups = [l for l, r in sorted(t.items()) if r["frac_positive"] >= 0.9 and r["p_sign"] < 0.01]
        add("A 팔이 상향되는 최소 lambda (subset32)", a["lambda_up_min"], ups[0] if ups else None,
            bool(ups) and ups[0] == a["lambda_up_min"], "동일")
        vr = min(r["var_ratio"] for r in t.values())
        add("A 팔 최소 분산비 (subset32)", a["var_ratio"], round(vr, 4),
            abs(math.log10(max(vr, 1e-9) / a["var_ratio"])) < 0.5, "같은 자릿수")
        bs = fast["baseline_subset32"]
        add("기준선 양의 기울기 비율 (subset32)", v1["baseline_frac_positive"],
            f"{bs['n_positive']}/{bs['n']}", None, "v1 도 GPU 에 따라 0.594 / 0.469 로 갈렸다")
        rs = fast.get("ref_subset32")
        if rs:
            k_ref = round(v1["ref_frac_positive"] * rs["n"])
            add("REF 양의 기울기 비율 (subset32)", v1["ref_frac_positive"],
                f"{rs['n_positive']}/{rs['n']}", abs(rs["n_positive"] - k_ref) <= REF_COUNT_TOL,
                f"+-{REF_COUNT_TOL}/32")
    return rows


def q1_select(summ: dict) -> tuple[str | None, str]:
    """outline 8.1 의 행 선택. 기준은 마지막 층·마지막 토큰의 비율이다."""
    pr, pi = summ["pretrained_over_random"], summ["pretrained_over_input"]
    if pr is None:
        return None, "무작위 초기화 대조군이 없어 선택할 수 없음"
    if pr < 1 / Q1_RATIO or (pr > Q1_RATIO and pi is not None and pi < 1 / Q1_RATIO):
        return None, NO_ROW
    if pr <= Q1_RATIO:
        return "like_random", S81["like_random"]
    key = "amplified" if (pi is not None and pi > Q1_RATIO) else "preserved"
    return key, S81[key]


def q5_select(fast: dict) -> tuple[str | None, str]:
    """outline 8.2 의 행 선택. spec 2.2 의 판정값을 그대로 쓴다."""
    v = fast["verdict"]
    if v in ("no_effect", "generic_perturbation"):
        return v, S82[v]
    if v == "direction_specific" and fast["ref_match"] is not None:
        key = "specific_like_ref" if fast["ref_match"] else "specific_unlike_ref"
        return key, S82[key]
    return None, NO_ROW


def main(args) -> int:
    out, tag = paths.out_root(), CFG.tag
    v1_all = load(paths.EXPECTED_DIR / "v1_baseline.json") or {}
    fps = load(paths.EXPECTED_DIR / "fingerprints.json") or {}
    man = load(out / "manifest.json") or {}

    doc = {"model": args.model, "noises_done": [], "q0": None, "q1": {}, "fast": {},
           "interpretation": {}, "q1_ratio": Q1_RATIO}
    md = [f"# Fast track 요약 — run `{man.get('run_id', out.name)}`", ""]

    data = {}
    for n in args.noise.split(","):
        rd = out / "results" / tag / f"{n}_{args.model}"
        s6 = load(rd / "stage6_summary.json")
        if s6 is None:
            continue
        data[n] = {"s3": load(rd / "stage3_summary.json"), "ctl": load(rd / "stage3_controls.json"),
                   "s4": load(rd / "stage4_summary.json"), "s6": s6,
                   "dmeta": load(out / "data" / tag / n / "meta.json")}
        doc["noises_done"].append(n)
    md += [f"모델 `{args.model}`, 끝난 노이즈: {doc['noises_done'] or '없음'}", ""]

    # --- Q0 ---------------------------------------------------------------
    md += ["## Q0. v1(Colab) OU 결과가 재현되는가", ""]
    if "ou" in data and "ou" in v1_all:
        d = data["ou"]
        rows = q0_rows(v1_all["ou"], d["dmeta"], d["s3"], d["s4"], d["s6"].get("fast"), fps.get("ou"))
        doc["q0"] = rows
        md += ["| 항목 | v1 | v2 | 판정 | 기준 (outline 6절) |", "|---|---|---|---|---|"]
        md += [f"| {r['item']} | {r['v1']} | {r['v2']} | {mark(r['ok'])} | {r['criterion']} |"
               for r in rows]
        bad = [r["item"] for r in rows if r["ok"] is False]
        md += ["", f"불일치 {len(bad)}건" + (f": {', '.join(bad)}" if bad else "") + ".",
               "" if args.model == "base" else "(v1 기준값은 Kronos-base 의 것이다. 다른 모델에서는 의미가 없다.)", ""]
    else:
        md += ["OU 가 아직 끝나지 않았다.", ""]

    for n, d in data.items():
        fast, ctl = d["s6"].get("fast"), d["ctl"]
        md += [f"## 노이즈 `{n}`", ""]

        # --- Q1 -----------------------------------------------------------
        md += ["### Q1. 선형 분리가 사전학습으로 생긴 것인가", ""]
        if ctl:
            s = ctl["summary"]
            key, sent = q1_select(s)
            doc["q1"][n] = {"summary": s, "key": key, "sentence": sent}
            b = s["band_mean"]
            md += ["| held-out LDR | 입력 기준선 | 무작위 (Kronos 초기화) | 무작위 (torch 기본, 참고) | 사전학습 "
                   "| 사전학습/무작위 | 사전학습/무작위(torch, 참고) | 사전학습/입력 |",
                   "|---|---|---|---|---|---|---|---|"]
            for label, d in ((f"layer {s['layer']}, t={s['t']}", s), (f"layer {s['layer']}, band 평균", b)):
                md.append(f"| {label} | {f(d['input'], '.2f')} | {f(d['random_init'], '.2f')} "
                          f"| {f(d.get('random_init_torch'), '.2f')} | {f(d['pretrained'], '.2f')} "
                          f"| {f(d['pretrained_over_random'], '.2f')} "
                          f"| {f(d.get('pretrained_over_random_torch'), '.2f')} "
                          f"| {f(d['pretrained_over_input'], '.2f')} |")
            md += ["", "판정에는 Kronos 자체 초기화 대조군의 비율(사전학습/무작위)만 쓴다. torch 기본 초기화는 "
                   "블록이 residual 에 거의 기여하지 않아 토큰 임베딩에 가까운 대조군이라 참고로만 둔다.", "",
                   f"해석 (outline 8.1, t={s['t']} 기준, 비율 {1/Q1_RATIO:g}~{Q1_RATIO:g} 을 '비슷'으로 봄): **{sent}**", ""]
        else:
            md += ["stage3_controls.json 이 없다.", ""]

        if not fast:
            md += ["fast 팔(A, J, L) 결과가 없다.", ""]
            continue
        doc["fast"][n] = {k: v for k, v in fast.items() if k != "table"}

        # --- Q2, Q3 -------------------------------------------------------
        md += ["### Q2·Q3. steering 효과가 S 방향에 특이적인가, 역방향도 작동하는가", "",
               "| λ_rel | A 양의 비율 (p) | A 분산비 | J seed별 양의 비율 | J 평균 | J 분산비 평균 | L 음의 비율 (p) | L 분산비 |",
               "|---|---|---|---|---|---|---|---|"]
        for r in fast["table"]:
            js = " / ".join(f"{s['frac_positive']:.0%}" for s in r["J"]["by_seed"])
            lcol = (f"{r['L']['frac_negative']:.1%} ({r['L']['p_sign_negative']:.1e}) | {r['L']['var_ratio']:.3f}"
                    if "L" in r else "- | -")
            md.append(f"| {r['lambda']} | {r['A']['frac_positive']:.1%} ({r['A']['p_sign']:.1e}) "
                      f"| {r['A']['var_ratio']:.3f} | {js} | {r['J']['frac_positive_mean']:.1%} "
                      f"| {r['J']['var_ratio_mean']:.3f} | {lcol} |")
        cos = [c for c in fast["table"][0]["J"]["cos_random_vs_S"] if c is not None]
        md += ["", f"표본 수 {fast['table'][0]['A']['n']}. 분산비는 표준편차의 비(개입 후 / 기준선). "
               + (f"J 의 |cos(R, S)| 레이어 평균: {', '.join(f'{c:.4f}' for c in cos)}." if cos else ""), ""]

        # --- Q4 -----------------------------------------------------------
        md += ["### Q4. 목표 분포(REF)와의 대조", ""]
        ref, star = fast.get("ref"), fast["lambda_star"]
        if ref:
            ks = next((r["A_vs_REF_ks_p"] for r in fast["table"] if r["lambda"] == star), None)
            md += [f"- REF(진짜 trend 입력, 개입 없음) 양의 기울기 비율: {ref['frac_positive']:.1%} "
                   f"({ref['n_positive']}/{ref['n']})",
                   f"- 기준선(base 입력, 개입 없음): {fast['baseline']['frac_positive']:.1%}",
                   f"- λ* = {star} 에서 A vs REF 의 KS p: {f(ks, '.2e')}", ""]
        else:
            md += ["REF 결과가 없다.", ""]

        # --- 판정 ---------------------------------------------------------
        key, sent = q5_select(fast)
        doc["interpretation"][n] = {"key": key, "sentence": sent}
        md += ["### 판정 (spec 2.2, 사전 고정)", "",
               "| A_up | λ* | 판정 | λ* 에서 J 평균 | collapse_generic | reversible | ref_match |",
               "|---|---|---|---|---|---|---|",
               f"| {fast['A_up']} | {star} | `{fast['verdict']}` | {f(fast['j_frac_positive_mean_at_star'], '.1%')} "
               f"| {fast['collapse_generic']} | {fast['reversible']} | {fast['ref_match']} |", "",
               f"해석 (outline 8.2): **{sent}**", ""]

    # --- 실행 메타 ---------------------------------------------------------
    env = man.get("env", {})
    doc["meta"] = {"code_commit": man.get("code_commit"), "gpu": env.get("gpu_name"),
                   "torch": env.get("torch"), "cuda": env.get("cuda"), "steps": man.get("steps", {})}
    md += ["## 실행 메타", "",
           f"- code commit `{man.get('code_commit')}`, plan `{man.get('plan')}`",
           f"- 모델 `{man.get('model_id')}` @ `{man.get('model_revision')}`",
           f"- GPU {env.get('gpu_name')} (driver {env.get('driver')}), torch {env.get('torch')}, CUDA {env.get('cuda')}",
           "- 단계별 소요: " + (", ".join(f"{k} {v:.0f}s" for k, v in man.get("steps", {}).items()) or "-"), ""]

    (out / "fast_summary.md").write_text("\n".join(md), encoding="utf-8")
    (out / "fast_summary.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print("\n".join(md))
    print(f"\n저장: {out / 'fast_summary.md'}\n      {out / 'fast_summary.json'}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 7 — fast track 요약")
    p.add_argument("--model", default="base", choices=["mini", "small", "base"])
    p.add_argument("--noise", default="ou,rw", help="요약에 넣을 노이즈 (끝난 것만 반영된다)")
    raise SystemExit(main(p.parse_args()))
