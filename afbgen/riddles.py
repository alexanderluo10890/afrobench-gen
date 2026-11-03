"""CulturalRiddles with the authors' own runner and scorer (slurm/riddles.sbatch).

    python -m afbgen.riddles shard   --raw DIR --shards 48 --out DIR [--limit N]
    python -m afbgen.riddles config  --model NAME --port P --dataset shard.jsonl --out cfg.yaml
    python -m afbgen.riddles merge   --raw DIR --out results.jsonl
    python -m afbgen.riddles summary --scored scored.parquet --model NAME --out summary.json

Parallelism: the runner sends one request at a time for vLLM (its batch mode does not run them
concurrently, and its cache client breaks in batch mode). So we split the canonical riddle file
into shards (rows unchanged) and run one unmodified runner per shard against the same server.
A restarted job skips riddles that already have an answer in <raw>/.

The run config is the authors' `benchmark_v0/qwen3.5_vllm.yaml` (riddle_master v1, riddle in its
source language, no hint, user message only, JSON mode, max 4096 tokens, no temperature sent, so
the model's own generation defaults apply) with only the model name, server URL and cache
changed. Scoring is the authors' `analysis.scoring` for new runs: rules, then the released first
human labels, then the released two-judge verdicts where both agree. No new (paid) judge calls.
Score 3 = correct; unscored answers are left out of accuracy (authors' definition).
"""

import argparse
import json
import os
from pathlib import Path

import yaml

from afbgen import models

AFRICAN = ["afaan_oromo_oromo", "arabic_egypt", "arabic_morocco", "luganda_uganda", "lumasaba_uganda",
           "swahili_kenya", "yoruba_nigeria"]


def riddles_dir() -> Path:
    return Path(os.environ["AFB_ROOT"]) / "CulturalRiddles"


def _rows(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def answered(raw: Path) -> dict[str, dict]:
    """riddle_id -> result row, from every results.jsonl under raw/ (rows with a model reply)."""
    done = {}
    for res in sorted(raw.glob("*/*/results.jsonl")):
        for r in _rows(res):
            if r.get("parsed_json") or (r.get("raw_response") or "").strip():
                done[r["riddle_id"]] = r
    return done


def make_shards(raw: Path, shards: int, out: Path, limit: int | None):
    rows = list(_rows(riddles_dir() / "data/canonical/riddles.jsonl"))[: limit or None]
    done = answered(raw)
    todo = [r for r in rows if r["riddle_id"] not in done]
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("shard*.jsonl"):
        old.unlink()
    n = 0
    for k in range(shards):
        part = todo[k::shards]
        if part:
            (out / f"shard{k:03d}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf-8")
            n += 1
    print(f"[shard] {len(rows)} riddles, {len(done)} already answered, {len(todo)} to do in {n} shards", flush=True)


def merge(raw: Path, out: Path):
    done = answered(raw)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in done.values()), encoding="utf-8")
    print(f"[merge] {len(done)} answered riddles -> {out}", flush=True)


def write_config(model: str, port: int, cache: Path, out: Path, dataset: Path | None = None):
    cfg = yaml.safe_load((riddles_dir() / "benchmark/configs/benchmark_v0/qwen3.5_vllm.yaml").read_text())
    m = models.get(model)
    cfg["base_url"] = f"http://127.0.0.1:{port}/v1"
    cfg["cache_dir"] = str(cache)
    cfg["cache_namespace"] = f"afbgen/canonical_v1/{model}/source_no_hint/master_v1"
    cfg["notes"] = f"afrobench-gen: {m['repo']}@{m['revision'][:10]} served by vLLM"
    cfg["model_spec"].update(model_name=model, model_id=model, provider_model_name=model,
                             notes=f"{m['repo']}@{m['revision']}")
    # One request at a time per runner, as in the authors' config; parallelism comes from shards.
    # No runner cache: resuming is done by make_shards (skips answered riddles).
    cfg["cache_enabled"] = False
    if dataset is not None:
        cfg["dataset_path"] = str(dataset)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def summarize(scored: Path, model: str, out: Path):
    import pandas as pd

    df = pd.read_parquet(scored)
    bench = {json.loads(l)["riddle_id"] for l in open(riddles_dir() / "data/canonical/evaluation_corpus.jsonl")}
    df = df[df["riddle_id"].isin(bench)]

    def stats(d):
        s = d["score"].dropna()
        return {"accuracy": round(100 * float((s == 3).mean()), 2) if len(s) else None,
                "scored": int(len(s)), "answers": int(len(d)),
                "coverage": round(100 * len(s) / max(1, len(d)), 1),
                "idk": round(100 * float((s == 1).mean()), 2) if len(s) else None}

    res = {"model": model, "all": stats(df), "african": stats(df[df["community"].isin(AFRICAN)]),
           "by_scorer": df["scorer"].value_counts().to_dict(),
           "by_community": {c: stats(g) for c, g in df.groupby("community")}}
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    a, f = res["all"], res["african"]
    print(f"[ok  ] culturalriddles {model}: accuracy {a['accuracy']} (coverage {a['coverage']}%), "
          f"African {f['accuracy']} (coverage {f['coverage']}%)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["shard", "config", "merge", "summary"])
    ap.add_argument("--model")
    ap.add_argument("--raw", type=Path)
    ap.add_argument("--shards", type=int, default=48)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--port", type=int)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--scored", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "shard":
        make_shards(a.raw, a.shards, a.out, a.limit)
    elif a.cmd == "merge":
        merge(a.raw, a.out)
    elif a.cmd == "config":
        write_config(a.model, a.port, a.cache, a.out, a.dataset)
    else:
        summarize(a.scored, a.model, a.out)


if __name__ == "__main__":
    main()
