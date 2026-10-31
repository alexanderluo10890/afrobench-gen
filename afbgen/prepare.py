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

_FLORES_FILES = sorted({langs.FLORES_PLUS_FILE[c] for c in langs.FLORES_LANGS + ["eng_Latn"]})

PATTERNS = {
    "afrimgsm": [f"data/{l}/test.tsv" for l in langs.TASK_LANGS["afrimgsm"]],
    "afrimmlu": [f"data/{l}/test.tsv" for l in langs.TASK_LANGS["afrimmlu"]],
    "afrixnli": [f"data/{l}/test.tsv" for l in langs.TASK_LANGS["afrixnli"]],
    "injongo": [f"{l}/test.jsonl" for l in langs.TASK_LANGS["injongo"]],
    "belebele": [f"data/{c}.jsonl" for c in langs.TASK_LANGS["belebele"]],
    "sib": [f"data/{c}/test.tsv" for c in langs.TASK_LANGS["sib"]],
    "flores": [f"devtest/{f}.jsonl" for f in _FLORES_FILES],
}


def data_root() -> Path:
    root = os.environ.get("AFB_DATA_DIR")
    if not root:
        raise SystemExit("set AFB_DATA_DIR (see slurm/env.sh)")
    return Path(root)


def _snapshot(repo, revision, local_dir, repo_type="model", patterns=None):
    # The marker records the revision AND the requested files, so adding languages
    # downloads the new files (snapshot_download only fetches what is missing).
    import hashlib

    key = revision if patterns is None else f"{revision} {hashlib.sha1(json.dumps(sorted(patterns)).encode()).hexdigest()[:12]}"
    marker = Path(local_dir) / ".afb_revision"
    if marker.exists() and marker.read_text().strip() == key:
        print(f"[skip] {repo}@{revision[:10]} already in {local_dir}")
        return
    print(f"[get ] {repo}@{revision[:10]} -> {local_dir}")
    snapshot_download(repo_id=repo, revision=revision, repo_type=repo_type,
                      local_dir=local_dir, allow_patterns=patterns)
    marker.write_text(key)


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
