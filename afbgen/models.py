"""Model registry (configs/models.yaml), tokenizer fix and tokenizer check.

The tokenizer check runs before every evaluation: the tokenizer that vLLM will load must give
the same token ids as the repository's own tokenizer.json. This catches bugs such as tiny-aya
under transformers v5 (CohereTokenizer splits digits one by one; tokenizer.json groups them
in threes; transformers issue #49176).

    python -m afbgen.models --check tiny-aya-earth      # on a login node, after prepare
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from afbgen import config

DEFAULT = "afriqueqwen-v2"

# digits (the tiny-aya bug), Latin with diacritics, Ethiopic, Arabic, mixed numbers/punctuation
CHECK_TEXTS = [
    "1234567 + 89 = 1234656. Janet sells 16 eggs at $2 each: 16 * 2 = 32.",
    "Ẹ kú àárọ̀. Ọmọ náà ní ẹ̀kọ́ 2024.",
    "ሰላም ነው? 3 ሰዎች 1,250 ብር ከፈሉ።",
    "مرحبا بكم في 2026، السعر 15.75 دولار.",
    "Habari yako? Bei ni shilingi 12,500 kwa kilo 3.",
    "Translate the following sentence from Hausa to English.\n\nHausa: Yanzu muna da beraye masu wata 4.",
    r"The answer is \boxed{70000}. Option (C) and 10^{-6}.",
]


def registry() -> dict:
    return config.load("models")


def get(name: str) -> dict:
    reg = registry()
    if name not in reg:
        sys.exit(f"unknown model '{name}'; known: {', '.join(reg)}")
    return reg[name]


def _data_root() -> Path:
    root = os.environ.get("AFB_DATA_DIR")
    if not root:
        raise SystemExit("set AFB_DATA_DIR (source slurm/env.sh)")
    return Path(root)


def model_dir(name: str) -> Path:
    """Download folder. Entries that share a repo (e.g. two revisions) must set `local_dir`."""
    m = get(name)
    return _data_root() / "models" / (m.get("local_dir") or m["repo"].split("/")[-1])


def tokenizer_path(name: str) -> Path:
    """The tokenizer the evaluation loads: the fixed copy if the model needs one."""
    return model_dir(name) / "tokenizer-fixed" if get(name).get("tokenizer_fix") else model_dir(name)


def apply_tokenizer_fix(name: str) -> None:
    """Copy the tokenizer files and set tokenizer_class to PreTrainedTokenizerFast, so that
    transformers builds the tokenizer from tokenizer.json without model-specific rules."""
    fix = get(name).get("tokenizer_fix")
    if not fix:
        return
    assert fix == "tokenizer_json", fix
    src, dst = model_dir(name), tokenizer_path(name)
    dst.mkdir(exist_ok=True)
    for f in ["tokenizer.json", "special_tokens_map.json", "chat_template.jinja", "generation_config.json"]:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    cfg = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    old = cfg.get("tokenizer_class")
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    (dst / "tokenizer_config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[fix ] {name}: tokenizer_class {old} -> PreTrainedTokenizerFast in {dst}")


def check_tokenizer(name: str) -> list[str]:
    """Problems found; empty list = the loaded tokenizer matches tokenizer.json."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    problems = []
    ref = Tokenizer.from_file(str(model_dir(name) / "tokenizer.json"))
    tok = AutoTokenizer.from_pretrained(str(tokenizer_path(name)))
    for text in CHECK_TEXTS:
        a = ref.encode(text, add_special_tokens=False).ids
        b = tok.encode(text, add_special_tokens=False)
        if a != b:
            problems.append(f"{type(tok).__name__} differs from tokenizer.json on {text[:40]!r}: "
                            f"{ref.encode(text, add_special_tokens=False).tokens[:12]} vs "
                            f"{tok.convert_ids_to_tokens(b)[:12]}")
    if getattr(tok, "chat_template", None) is None:
        problems.append("no chat template")
    else:
        rendered = tok.apply_chat_template([{"role": "user", "content": "2 + 2 = ?"}], tokenize=False,
                                           add_generation_prompt=True)
        if "2 + 2 = ?" not in rendered:
            problems.append("chat template does not contain the user message")
    return problems


def describe(name: str) -> None:
    """Facts to record before a run: context length, architecture, recent commits."""
    cfg = json.loads((model_dir(name) / "config.json").read_text(encoding="utf-8"))
    text = cfg.get("text_config", cfg)
    print(f"[info] {name}: arch={cfg.get('architectures')} max_position_embeddings="
          f"{text.get('max_position_embeddings')} vision={'vision_config' in cfg}")
    try:
        from huggingface_hub import list_repo_commits

        for c in list_repo_commits(get(name)["repo"])[:5]:
            print(f"        commit {c.commit_id[:10]} {c.created_at:%Y-%m-%d} {c.title[:70]}")
    except Exception as e:  # offline or no access: not fatal
        print(f"        (commit history unavailable: {type(e).__name__})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs="+", metavar="MODEL", help="model names, or 'all'")
    a = ap.parse_args()
    names = list(registry()) if a.check == ["all"] else a.check
    bad = 0
    for n in names:
        if not (model_dir(n) / "tokenizer.json").exists():
            print(f"[skip] {n}: not downloaded")
            continue
        describe(n)
        p = check_tokenizer(n)
        bad += bool(p)
        print(f"[{'ok  ' if not p else 'FAIL'}] {n}: tokenizer " + ("matches tokenizer.json" if not p else ""))
        for x in p:
            print("        " + x)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
