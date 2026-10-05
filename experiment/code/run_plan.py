#!/usr/bin/env python3
"""파이프라인 러너 — 1차 실험의 colab_control.py 를 대체한다 (spec 3.11).

    python experiment/code/run_plan.py --plan fast  --run-id v2_fast     --model base
    python experiment/code/run_plan.py --plan smoke --run-id local_smoke --model small

fast  : 노이즈 X in [ou, rw] 순서로 Stage 1~6 + 대조군. 노이즈 하나가 끝날 때마다 요약을
        만들고(stage7) 결과 브랜치에 중간 push 한다 — RW 가 도는 동안 OU 결과를 먼저 본다.
smoke : ou 만, 각 stage 의 --smoke 경로. 환경·경로·훅 오류를 몇 분 안에 잡는 것이 목적이다.

출력 루트 (stage 에는 환경변수로 전달된다, kexp/paths.py)
    KEXP_OUT      없으면 experiment/runs/<run-id>
    KEXP_SCRATCH  없으면 pod 에서는 /root/kexp_scratch/<run-id>, 로컬에서는 experiment/_scratch/<run-id>

각 단계는 stage 스크립트를 subprocess 로 실행하고 로그를 <OUT>/logs/<step>.log 에 남긴다.
실패하면 즉시 중단하고 그 종료 코드를 돌려준다. 단계가 끝나면 <OUT>/.done/<step> 마커를
남기고, 같은 run-id 로 다시 실행하면 마커가 있는 단계는 건너뛴다.

실험 로직은 여기에 넣지 않는다. 실행 조건은 stage 스크립트의 인자로만 표현한다.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
REPO = CODE.parents[1]
sys.path.insert(0, str(CODE))

from kexp import paths, runmeta  # noqa: E402
from kexp.config import CFG  # noqa: E402

RUNNER_VERSION = "run_plan v2-fast"   # 실행 시 출력된다. 이 값이 다르면 옛 러너다.
STAGES = {
    "stage0": "stage0_smoke.py", "stage1": "stage1_dataset.py", "stage2": "stage2_activations.py",
    "stage3": "stage3_ldr.py", "stage4": "stage4_steering_vector.py",
    "stage5": "stage5_intervene.py", "stage6": "stage6_report.py",
    "stage7": "stage7_fast_summary.py",
}


def build_steps(plan: str, model: str, noises: list[str]) -> list[dict]:
    """(이름, stage, 인자) 목록. spec 3.11 의 표와 같은 순서다."""
    smoke = ["--smoke"] if plan == "smoke" else []
    n_eval = [] if smoke else ["--n-eval", str(CFG.steer.n_eval_fast),
                               "--eval-batch", str(CFG.steer.eval_batch)]
    steps = [dict(name="00_stage0", stage="stage0", args=[])]
    for x in noises:
        nz = ["--noise", x]
        steps += [
            dict(name=f"{x}_01_stage1", stage="stage1", args=nz + ["--expect-fingerprint"]),
            dict(name=f"{x}_02_stage2", stage="stage2", args=nz + smoke, scratch=x),
            dict(name=f"{x}_03_stage2_randinit", stage="stage2", scratch=x + "_randinit",
                 args=nz + ["--random-init", "--positions", "bands"] + smoke),
            dict(name=f"{x}_03b_stage2_randinit_torch", stage="stage2", scratch=x + "_randinit_torch",
                 args=nz + ["--random-init", "--init", "torch", "--positions", "bands"] + smoke),
            dict(name=f"{x}_04_stage3", stage="stage3", args=nz + ["--controls"] + smoke),
            dict(name=f"{x}_05_stage4", stage="stage4", args=nz + smoke),
            dict(name=f"{x}_06_clean_scratch", clean=x),
            dict(name=f"{x}_07_stage5_fast", stage="stage5",
                 args=nz + ["--arms", "fast", "--lambdas", "fast"] + n_eval + smoke),
            dict(name=f"{x}_08_stage5_ref", stage="stage5",
                 args=nz + ["--reference-trend"] + n_eval + smoke),
            dict(name=f"{x}_09_stage6", stage="stage6", args=nz + smoke),
            dict(name=f"{x}_10_summary", stage="stage7", args=["--noise", ",".join(noises)],
                 push=True),
        ]
    for s in steps:
        if s.get("stage") not in (None, "stage1"):      # stage1 은 모델과 무관하다
            s["args"] = ["--model", model] + s["args"]
    return steps


def activation_dirs(scratch: Path, noise: str) -> list[Path]:
    base = scratch / "activations" / CFG.tag
    return [base / noise, base / f"{noise}_randinit", base / f"{noise}_randinit_torch"]


def run_step(cmd: list[str], env: dict, log_path: Path) -> int:
    """출력을 화면과 로그 파일에 동시에 흘려보낸다."""
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n##### {runmeta.now_utc()} $ {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        proc.wait()
    return proc.returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True, choices=["smoke", "fast"])
    p.add_argument("--run-id", required=True)
    p.add_argument("--model", default="base", choices=["base", "small"])
    p.add_argument("--noise", default=None, help="쉼표로 구분 (기본: fast = ou,rw / smoke = ou)")
    args = p.parse_args()
    noises = (args.noise.split(",") if args.noise else (["ou"] if args.plan == "smoke" else ["ou", "rw"]))

    on_pod = bool(os.environ.get("RUNPOD_POD_ID"))
    out = Path(os.environ.get("KEXP_OUT") or paths.RUNS_DIR / args.run_id).resolve()
    scratch = Path(os.environ.get("KEXP_SCRATCH") or (
        f"/root/kexp_scratch/{args.run_id}" if on_pod else REPO / "experiment" / "_scratch" / args.run_id))
    env = dict(os.environ, KEXP_OUT=str(out), KEXP_SCRATCH=str(scratch), PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join(filter(None, [str(CODE), os.environ.get("PYTHONPATH")])))
    os.environ.update(KEXP_OUT=str(out), KEXP_SCRATCH=str(scratch))
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / ".done").mkdir(exist_ok=True)

    steps = build_steps(args.plan, args.model, noises)
    print(f"[{RUNNER_VERSION}] run_id {args.run_id} | plan {args.plan} | model {args.model} | "
          f"noise {noises} | commit {(runmeta.code_commit() or 'unknown')[:12]}")
    print(f"  OUT     : {out}\n  SCRATCH : {scratch}")

    try:      # 재개: 앞 세션의 단계별 소요 시간을 이어 쓴다
        timings = json.loads((out / "manifest.json").read_text()).get("steps", {})
    except (OSError, ValueError):
        timings = {}
    man = dict(noises=noises, steps=timings, status="running")
    runmeta.write_manifest(out, args.run_id, args.plan, args.model, **man)
    branch = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    t_all = time.time()
    for i, s in enumerate(steps, 1):
        marker = out / ".done" / s["name"]
        done = marker.exists()
        if done and "scratch" in s:
            # 활성화는 컨테이너 디스크에 있어 pod 가 바뀌면 사라진다. 마커가 있어도 이후 단계가
            # 아직 읽어야 하는데 파일이 없으면 다시 추출한다.
            cleaned = (out / ".done" / f"{s['name'].split('_')[0]}_06_clean_scratch").exists()
            have = any((scratch / "activations" / CFG.tag / s["scratch"]).glob("*/base/meta.json"))
            done = cleaned or have
        if done:
            print(f"\n[{i:02d}/{len(steps)}] {s['name']}: 이미 끝남 (.done) — 건너뜀")
            continue
        print(f"\n{'=' * 70}\n[{i:02d}/{len(steps)}] {s['name']}\n{'=' * 70}", flush=True)
        t0 = time.time()
        if "clean" in s:
            for d in activation_dirs(scratch, s["clean"]):
                shutil.rmtree(d, ignore_errors=True)
                print(f"  scratch 정리: {d}")
            rc = 0
        else:
            cmd = [sys.executable, "-u", str(CODE / "stages" / STAGES[s["stage"]]), *s["args"]]
            rc = run_step(cmd, env, out / "logs" / f"{s['name']}.log")
        timings[s["name"]] = round(time.time() - t0, 1)
        if rc != 0:
            print(f"\n{s['name']} 실패 (exit {rc}). 이후 단계는 실행하지 않는다. "
                  f"로그: {out / 'logs' / (s['name'] + '.log')}")
            runmeta.write_manifest(out, args.run_id, **dict(man, status="failed", failed_step=s["name"]))
            return rc
        marker.write_text(runmeta.now_utc() + "\n")
        runmeta.write_manifest(out, args.run_id, **man)
        print(f"-- {s['name']} 완료 ({timings[s['name']]}s, 누적 {(time.time() - t_all) / 60:.1f}분)")
        if s.get("push") and args.plan == "fast" and branch.startswith("results/"):
            # 노이즈 하나가 끝났다. 요약과 그림을 결과 브랜치에 올린다. 실패해도 계속 간다.
            rc_push = subprocess.run(["bash", "RunPod/push_meta.sh", str(out)], cwd=REPO).returncode
            print(f"-- 중간 push {'완료' if rc_push == 0 else '실패 (계속 진행)'}")

    runmeta.write_manifest(out, args.run_id, **dict(man, status="ok", finished_utc=runmeta.now_utc()))
    print(f"\n[{RUNNER_VERSION}] 전체 완료: {len(steps)}단계, {(time.time() - t_all) / 60:.1f}분. "
          f"요약: {out / 'fast_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
