from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "tasks" / "afrobench_gen"


def load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))


def math_instruction(lang: str) -> str:
    """AfriMGSM instruction for a dataset language code (note, Appendix C)."""
    from afbgen import langs

    return load("math_instructions")[langs.TO_FLORES[lang]]
