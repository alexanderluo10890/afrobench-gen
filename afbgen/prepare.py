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

from afbgen import config, langs, models

_FLORES_FILES = sorted({langs.FLORES_PLUS_FILE[c] for c in langs.FLORES_LANGS + ["eng_Latn"]})

PATTERNS = {
    "afrimgsm": [f"data/{l}/test.tsv" for l in langs.TASK_LANGS["afrimgsm"]],
    "afrimmlu": [f"data/{l}/test.tsv" for l in langs.TASK_LANGS["afrimmlu"]],
    "afrixnli": [f"data/{l}/test.tsv" for l in langs.TASK_LANGS["afrixnli"]],
    "injongo": [f"{l}/test.jsonl" for l in langs.TASK_LANGS["injongo"]],
    "belebele": [f"data/{c}.jsonl" for c in langs.TASK_LANGS["belebele"]],
    "sib": [f"data/{c}/test.tsv" for c in langs.TASK_LANGS["sib"]],
    "flores": [f"devtest/{f}.jsonl" for f in _FLORES_FILES],
    "globalpiqa": [f"data/nonparallel_{l}.tsv" for l in langs.TASK_LANGS["globalpiqa"]],
    "mmluprox": [f"{l}/test-*.parquet" for l in langs.TASK_LANGS["mmluprox"]],
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
    prepare_github(pins)


def prepare_github(pins):
    """mHumanEval Python files from GitHub at the pinned commit (one JSON file per language)."""
    import urllib.parse
    import urllib.request

    g = pins["github"]["mhumaneval"]
    d = data_root() / "datasets" / "mhumaneval"
    d.mkdir(parents=True, exist_ok=True)
    marker = d / ".afb_revision"
    key = f"{g['commit']} {len(langs.TASK_LANGS['mhumaneval'])}"
    if marker.exists() and marker.read_text().strip() == key:
        print(f"[skip] {g['repo']}@{g['commit'][:10]} already in {d}")
        return
    print(f"[get ] {g['repo']}@{g['commit'][:10]} -> {d}")
    for lang in langs.TASK_LANGS["mhumaneval"]:
        path = urllib.parse.quote(g["path"].format(lang=lang))
        url = f"https://raw.githubusercontent.com/{g['repo']}/{g['commit']}/{path}"
        data = urllib.request.urlopen(url, timeout=60).read()
        assert len(json.loads(data)) == 164, (lang, url)
        (d / f"{lang}.json").write_bytes(data)
    marker.write_text(key)


def prepare_model(pins, names=(models.DEFAULT,)):
    """Download each model at its pinned revision, fix its tokenizer if needed, check the tokenizer."""
    for name in names:
        m = models.get(name)
        marker = models.model_dir(name) / ".afb_revision"
        if not (marker.exists() and marker.read_text().strip() == m["revision"]):
            print(f"[get ] {m['repo']}@{m['revision'][:10]} -> {models.model_dir(name)}")
            snapshot_download(repo_id=m["repo"], revision=m["revision"], local_dir=models.model_dir(name),
                              ignore_patterns=["*.gguf", "original/*", "*.pth", "*.msgpack", "*.h5"])
            marker.write_text(m["revision"])
        else:
            print(f"[skip] {m['repo']}@{m['revision'][:10]} already in {models.model_dir(name)}")
        models.apply_tokenizer_fix(name)
        models.describe(name)
        problems = models.check_tokenizer(name)
        print(f"[{'ok  ' if not problems else 'FAIL'}] {name}: tokenizer "
              + ("matches tokenizer.json" if not problems else "DOES NOT match tokenizer.json"))
        for x in problems:
            print("        " + x)


def prepare_comet(pins):
    c = pins["comet"]
    for name, v in c["variants"].items():
        _snapshot(v["repo"], v["revision"], data_root() / "models" / f"ssa-comet-{name}")
    for repo, rev in sorted({(v["encoder_repo"], v["encoder_revision"]) for v in c["variants"].values()}):
        _cache_encoder(repo, rev)


def _cache_encoder(repo: str, rev: str):
    """COMET rebuilds its encoder from the hub id stored in model.ckpt (at "main"), so put the
    pinned encoder in the HF cache and point refs/main at it for offline use. A correct cache
    is left untouched: rewriting it while jobs run can make files briefly invisible to compute
    nodes on network storage."""
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(repo, "tokenizer.json", revision=rev)
    if isinstance(cached, str):
        refs = Path(cached).parents[2] / "refs"
        if (refs / "main").exists() and (refs / "main").read_text().strip() == rev:
            print(f"[skip] encoder {repo}@{rev[:10]} already in HF cache")
            return
    path = snapshot_download(repo_id=repo, revision=rev, ignore_patterns=["events.out.tfevents*", "*.msgpack"])
    refs = Path(path).parents[1] / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(rev)
    print(f"[ok  ] encoder {repo}@{rev[:10]} in HF cache")


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
    ap.add_argument("--model", default=models.DEFAULT,
                    help="comma-separated names from configs/models.yaml, or 'all' (used by the model step)")
    a = ap.parse_args()
    pins = config.load("pins")
    steps = {"data": prepare_data, "model": prepare_model, "comet": prepare_comet, "verify": verify}
    names = list(models.registry()) if a.model == "all" else a.model.split(",")
    for name, fn in steps.items():
        if a.only in (None, name):
            fn(pins, names) if name == "model" else fn(pins)


if __name__ == "__main__":
    main()
