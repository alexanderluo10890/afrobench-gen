from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "tasks" / "afrobench_gen"


def load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))


def math_system_prompt(lang: str) -> str:
    from afbgen import langs

    cfg = load("math_system_prompts")
    if lang in cfg["overrides"]:
        return cfg["overrides"][lang]
    return cfg["default_template"].format(language=langs.name(lang))
