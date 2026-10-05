"""pytest 공용 설정 (spec 6.1).

    ~/.venvs/kronos/bin/python -m pytest experiment/code/tests -q

모델 가중치를 내려받지 않는다. 아주 작은 Kronos 를 직접 만들거나 가짜 모듈을 쓴다.
"""

import importlib.util
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))


def load_stage(name: str):
    """stages/ 는 패키지가 아니라 실행 스크립트 모음이다. 파일 경로로 불러온다."""
    path = next((CODE / "stages").glob(f"{name}_*.py"))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
