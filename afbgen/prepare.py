"""Download the model, SSA-COMET and all datasets at their pinned revisions.

Run once per cluster on a node with internet (login node), before submitting jobs:

    python -m afbgen.prepare            # everything
    python -m afbgen.prepare --only data

Files go to $AFB_DATA_DIR (datasets/, models/) and $HF_HOME (COMET encoder).
Compute jobs then run fully offline (HF_HUB_OFFLINE=1).
"""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from afbgen import config, langs

FLORES_CODES = sorted({langs.flores_code(l) for l in langs.LANGS})
MASAKHANE = sorted(set(langs.TASK_LANGS["afrimgsm"]))

PATTERNS = {
    "afrimgsm": [f"data/{l}/test.tsv" for l in MASAKHANE],
    "afrimmlu": [f"data/{l}/test.tsv" for l in MASAKHANE],
    "afrixnli": [f"data/{l}/test.tsv" for l in MASAKHANE],
    "injongo": [f"{l}/test.jsonl" for l in MASAKHANE],
    "belebele": [f"data/{c}.jsonl" for c in FLORES_CODES],
    "sib": [f"data/{c}/test.tsv" for c in FLORES_CODES],
    "flores": [f"data/language/{c}/devtest-*.parquet" for c in FLORES_CODES],
}


def data_root() -> Path:
    root = os.environ.get("AFB_DATA_DIR")
    if not root:
        raise SystemExit("set AFB_DATA_DIR (see slurm/env.sh)")
    return Path(root)


def _snapshot(repo, revision, local_dir, repo_type="model", patterns=None):
    marker = Path(local_dir) / ".afb_revision"
    if marker.exists() and marker.read_text().strip() == revision:
        print(f"[skip] {repo}@{revision[:10]} already in {local_dir}")
        return
    print(f"[get ] {repo}@{revision[:10]} -> {local_dir}")
    snapshot_download(repo_id=repo, revision=revision, repo_type=repo_type,
                      local_dir=local_dir, allow_patterns=patterns)
    marker.write_text(revision)


def prepare_data(pins):
    for task, pin in pins["datasets"].items():
        _snapshot(pin["repo"], pin["revision"], data_root() / "datasets" / task, "dataset", PATTERNS[task])


def prepare_model(pins):
    m = pins["model"]
    _snapshot(m["repo"], m["revision"], model_dir(pins))


def prepare_comet(pins):
    c = pins["comet"]
    for name, v in c["variants"].items():
        _snapshot(v["repo"], v["revision"], data_root() / "models" / f"ssa-comet-{name}")
    # COMET rebuilds its encoder from the hub id in hparams.yaml (at "main"), so put the
    # pinned encoder in the HF cache and point refs/main at it for offline use.
    path = snapshot_download(repo_id=c["encoder_repo"], revision=c["encoder_revision"])
    refs = Path(path).parents[1] / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(c["encoder_revision"])
    print(f"[ok  ] encoder {c['encoder_repo']}@{c['encoder_revision'][:10]} in HF cache")


def model_dir(pins) -> Path:
    return data_root() / "models" / pins["model"]["repo"].split("/")[-1]


def verify(pins):
    """Load every (task, lang) once and print sizes; fails loudly on missing files."""
    from afbgen import tasklib

    sizes = {}
    for task in langs.TASKS:
        for lang in langs.TASK_LANGS[task]:
            sizes[f"{task}/{lang}"] = len(tasklib.load_rows(task, lang))
    print(json.dumps(sizes, indent=1))
    return sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["data", "model", "comet", "verify"])
    a = ap.parse_args()
    pins = config.load("pins")
    steps = {"data": prepare_data, "model": prepare_model, "comet": prepare_comet, "verify": verify}
    for name, fn in steps.items():
        if a.only in (None, name):
            fn(pins)


if __name__ == "__main__":
    main()
