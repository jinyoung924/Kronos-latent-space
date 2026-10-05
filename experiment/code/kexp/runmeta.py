"""실행 메타데이터 — manifest, 체크섬, 구조 검사 (spec 3.13).

Kronos-investing 의 checksum.py / verify_run.py / env_info.py 역할을 하나로 합친다.

    PYTHONPATH=experiment/code python -m kexp.runmeta write-manifest  --run-id ID [--plan fast]
    PYTHONPATH=experiment/code python -m kexp.runmeta checksum-write  --run-id ID    # pod, 실행이 끝난 뒤
    PYTHONPATH=experiment/code python -m kexp.runmeta checksum-verify --run-id ID    # 로컬, fetch 뒤. 불일치면 exit 1
    PYTHONPATH=experiment/code python -m kexp.runmeta verify-run      --run-id ID --plan fast

실행의 출력 루트 <OUT> 은 환경변수 KEXP_OUT, 없으면 experiment/runs/<run-id> 다.
checksum-verify 는 로컬에서 venv 없이 돌 수 있어야 하므로 torch 는 필요할 때만 임포트한다.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from kexp import paths

# 체크섬에서 빼는 것: 실행 중 계속 쓰이는 로그, 재개 마커, 체크섬 파일 자신
CHECKSUM_SKIP_DIRS = ("logs", ".done")
CHECKSUM_FILE = "checksums.json"


def out_dir(run_id: str) -> Path:
    v = os.environ.get("KEXP_OUT", "").strip()
    return Path(v) if v else paths.RUNS_DIR / run_id


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _run(*cmd: str) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, cwd=paths.REPO_ROOT)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def code_commit() -> str | None:
    """코드를 마지막으로 바꾼 커밋. 결과 브랜치의 커밋은 experiment/runs 만 건드리므로 뺀다."""
    return os.environ.get("CODE_COMMIT") or _run("git", "log", "-1", "--format=%H", "--", ".",
                                                 ":!experiment/runs")


def env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "hostname": platform.node(), "pod_id": os.environ.get("RUNPOD_POD_ID"),
            "torch": None, "cuda": None, "gpu_name": None, "driver": None}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    smi = _run("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader")
    if smi:
        name, _, driver = smi.splitlines()[0].partition(",")
        info["gpu_name"] = info["gpu_name"] or name.strip()
        info["driver"] = driver.strip()
    return info


def write_manifest(out: Path, run_id: str, plan: str | None = None, model: str | None = None,
                   **extra) -> Path:
    """manifest.json 을 만들거나 갱신한다. extra 는 러너가 넘기는 값(noises, steps, 시각)이다."""
    import dataclasses

    from kexp.config import CFG, resolve_model

    f = out / "manifest.json"
    try:
        man = json.loads(f.read_text())
    except (OSError, ValueError):
        man = {"started_utc": now_utc()}
    plan = plan or man.get("plan")
    model = model or man.get("model") or "base"
    cfg = dataclasses.replace(CFG, model=resolve_model(model, CFG.model.max_context))
    man.update({
        "run_id": run_id, "plan": plan, "model": model,
        "code_commit": code_commit(),
        "kronos_dir_commit": _run("git", "log", "-1", "--format=%H", "--", "Kronos"),
        "model_id": cfg.model.model_id, "model_revision": cfg.model.revision,
        "tokenizer_id": cfg.model.tokenizer_id, "tokenizer_revision": cfg.model.tokenizer_revision,
        "config": cfg.to_dict(), "config_hash": cfg.hash(),
        "env": env_info(), "updated_utc": now_utc(),
    })
    man.update(extra)
    out.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    return f


# --- 체크섬 -------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(out: Path) -> dict:
    files = {}
    for f in sorted(out.rglob("*")):
        rel = f.relative_to(out)
        if not f.is_file() or rel.parts[0] in CHECKSUM_SKIP_DIRS or rel.as_posix() == CHECKSUM_FILE:
            continue
        files[rel.as_posix()] = {"bytes": f.stat().st_size, "sha256": _sha256(f)}
    return files


def checksum_write(out: Path, run_id: str) -> int:
    entries = scan(out)
    if not entries:
        print(f"no files under {out}")
        return 1
    doc = {"run_id": run_id, "n_files": len(entries),
           "total_bytes": int(sum(e["bytes"] for e in entries.values())),
           "code_commit": code_commit(), "files": entries}
    (out / CHECKSUM_FILE).write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"wrote {out / CHECKSUM_FILE}: {doc['n_files']} files, {doc['total_bytes']:,} bytes")
    return 0


def checksum_verify(out: Path) -> int:
    f = out / CHECKSUM_FILE
    if not f.exists():
        print(f"missing {f}")
        return 1
    doc = json.loads(f.read_text())
    actual = scan(out)
    problems = []
    for name, e in doc["files"].items():
        a = actual.get(name)
        if a is None:
            problems.append(f"MISSING  {name}")
        elif a["bytes"] != e["bytes"]:
            problems.append(f"SIZE     {name}: expected {e['bytes']:,}, found {a['bytes']:,}")
        elif a["sha256"] != e["sha256"]:
            problems.append(f"SHA256   {name}: content differs")
    problems += [f"UNLISTED {name}" for name in actual if name not in doc["files"]]
    if len(actual) != doc["n_files"]:
        problems.append(f"COUNT    expected {doc['n_files']} files, found {len(actual)}")
    if problems:
        print("\n".join(problems))
        print(f"FAIL: {len(problems)} problem(s) in {out}")
        return 1
    print(f"OK: {doc['n_files']} files match {f}")
    return 0


# --- 구조 검사 ----------------------------------------------------------------

def verify_run(out: Path, plan: str) -> int:
    """계획이 남겼어야 할 파일이 있는지, Stage 5 결과가 완전한지 확인한다."""
    from kexp.config import CFG

    try:
        man = json.loads((out / "manifest.json").read_text())
    except (OSError, ValueError):
        man = {}
    model = man.get("model") or "base"
    smoke = plan == "smoke"
    noises = man.get("noises") or (["ou"] if smoke else ["ou", "rw"])
    tag = CFG.tag
    # fast: (A + J x seed 수 + L) x lambda 격자 + REF 1 = 5 x 5 + 1 = 26.  smoke: 3 x 2 + 1 = 7
    n_keys = (3 * 2 if smoke else (2 + len(CFG.control.random_seeds)) * len(CFG.steer.lambdas_rel_fast)) + 1
    steer = "steer/results_smoke.json" if smoke else "steer/results.json"

    problems = []
    for n in noises:
        need = [f"data/{tag}/{n}/base.npz", f"data/{tag}/{n}/trend.npz"]
        rd = f"results/{tag}/{n}_{model}"
        need += [f"{rd}/{x}" for x in ("ldr.npz", "stage3_summary.json", "stage3_controls.json",
                                        "steering.npz", "stage4_summary.json", steer,
                                        "stage6_summary.json")]
        problems += [f"MISSING  {p}" for p in need if not (out / p).is_file()]
        if (out / rd / steer).is_file():
            res = json.loads((out / rd / steer).read_text())
            if len(res) != n_keys:
                problems.append(f"KEYS     {rd}/{steer}: expected {n_keys} entries, found {len(res)}")
            bad = [k for k, v in res.items() if any(s != s for s in v.get("slopes", [float("nan")]))]
            if bad:
                problems.append(f"NAN      {rd}/{steer}: NaN or missing slopes in {bad[:5]}")
    problems += [f"MISSING  {p}" for p in ("fast_summary.md", "fast_summary.json")
                 if not (out / p).is_file()]
    if problems:
        print("\n".join(problems))
        print(f"FAIL: {len(problems)} problem(s) in {out} (plan {plan}, noises {noises})")
        return 1
    print(f"OK: run {out.name} is complete (plan {plan}, model {model}, noises {noises})")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["write-manifest", "checksum-write", "checksum-verify", "verify-run"])
    p.add_argument("--run-id", required=True)
    p.add_argument("--plan", default=None, choices=["fast", "smoke"])
    p.add_argument("--model", default=None)
    a = p.parse_args(argv)
    out = out_dir(a.run_id)
    if a.action == "write-manifest":
        print(write_manifest(out, a.run_id, a.plan, a.model))
        return 0
    if a.action == "checksum-write":
        return checksum_write(out, a.run_id)
    if a.action == "checksum-verify":
        return checksum_verify(out)
    return verify_run(out, a.plan or "fast")


if __name__ == "__main__":
    sys.exit(main())
