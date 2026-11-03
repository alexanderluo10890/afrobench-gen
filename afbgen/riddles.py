"""CulturalRiddles with the authors' own runner and scorer (slurm/riddles.sbatch).

    python -m afbgen.riddles config  --model NAME --port P --cache DIR --out cfg.yaml
    python -m afbgen.riddles summary --scored scored.parquet --model NAME --out summary.json

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


def write_config(model: str, port: int, cache: Path, out: Path):
    cfg = yaml.safe_load((riddles_dir() / "benchmark/configs/benchmark_v0/qwen3.5_vllm.yaml").read_text())
    m = models.get(model)
    cfg["base_url"] = f"http://127.0.0.1:{port}/v1"
    cfg["cache_dir"] = str(cache)
    cfg["cache_namespace"] = f"afbgen/canonical_v1/{model}/source_no_hint/master_v1"
    cfg["notes"] = f"afrobench-gen: {m['repo']}@{m['revision'][:10]} served by vLLM"
    cfg["model_spec"].update(model_name=model, model_id=model, provider_model_name=model,
                             notes=f"{m['repo']}@{m['revision']}")
    # Throughput only: concurrent requests through the runner's vLLM batch path (vllm_batch.py).
    # Same prompt and request body per riddle; the authors' example sends them one by one.
    cfg["batch_mode"], cfg["batch_size"] = True, 64
    cfg["model_spec"]["capabilities"]["supports_batch"] = True
    # The runner's cache client passes batch_size to AsyncOpenAI, which rejects it, so batch mode
    # only works without the cache. The cache affects resuming only, not the answers.
    cfg["cache_enabled"] = False
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
    ap.add_argument("cmd", choices=["config", "summary"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--scored", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "config":
        write_config(a.model, a.port, a.cache, a.out)
    else:
        summarize(a.scored, a.model, a.out)


if __name__ == "__main__":
    main()
