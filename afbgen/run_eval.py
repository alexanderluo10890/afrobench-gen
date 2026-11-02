"""Run one shard: one benchmark, a list of languages, one model load.

    python -m afbgen.run_eval --run-dir $AFB_RESULTS/<run> --task afrimmlu --langs eng,hau,ibo

Durability (jobs may be preempted or hit the 3 h limit at any moment):
  * every generation is committed to <run>/<task>/<lang>/cache_rank0.db (SQLite,
    autocommit) as soon as its vLLM chunk finishes; a restarted job replays the cache and
    only generates what is missing;
  * a language is finished when results.json exists (written atomically, last);
    finished languages are skipped on restart.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from afbgen import INFRA_VERSION, config, langs, models

# --------------------------------------------------------------------------- provenance

def git_state() -> dict:
    def g(*a):
        return subprocess.run(["git", *a], cwd=config.ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": g("rev-parse", "HEAD"), "dirty": bool(g("status", "--porcelain")),
            "tag": g("describe", "--tags", "--always")}


def provenance(allow_dirty: bool) -> dict:
    gs = git_state()
    if not gs["commit"]:
        sys.exit("not a git checkout: results must be traceable to a commit")
    if gs["dirty"] and not allow_dirty:
        sys.exit("working tree has uncommitted changes; commit them (and bump INFRA_VERSION if "
                 "they affect scores) or pass --allow-dirty for a throwaway run")
    import importlib.metadata as md

    vers = {}
    for p in ("lm_eval", "vllm", "torch", "transformers", "datasets", "sacrebleu"):
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    return {"infra_version": INFRA_VERSION, "git": gs, "packages": vers, "pins": config.load("pins"),
            "generation": config.load("generation"), "math_instructions": config.load("math_instructions")}


def lock_run_manifest(run_dir: Path, prov: dict, model_name: str):
    """All shards of a run must use identical code/config and the same model. First shard writes run.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.json"
    key = {k: prov[k] for k in ("infra_version", "pins", "generation", "math_instructions")}
    key["commit"] = prov["git"]["commit"]
    key["model"] = {"name": model_name, **models.get(model_name)}
    try:
        with open(path, "x", encoding="utf-8") as f:
            json.dump(key, f, indent=1)
    except FileExistsError:
        have = json.loads(path.read_text(encoding="utf-8"))
        if have != json.loads(json.dumps(key)):
            sys.exit(f"{path} was produced by a different commit/config; use a new --run-dir")


def check_gpu(allow_any: bool) -> str:
    import torch

    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    if "L40S" not in name and not allow_any:
        sys.exit(f"GPU is '{name}'; this evaluation must run on an L40S")
    return name


def check_model(model_name: str):
    """Pinned revision on disk, and a tokenizer that matches the repo's tokenizer.json."""
    marker = models.model_dir(model_name) / ".afb_revision"
    if not marker.exists() or marker.read_text().strip() != models.get(model_name)["revision"]:
        sys.exit(f"{models.model_dir(model_name)} is not the pinned revision; "
                 f"run `python -m afbgen.prepare --only model --model {model_name}`")
    problems = models.check_tokenizer(model_name)
    if problems:
        sys.exit(f"tokenizer check failed for {model_name}:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------------------- model

def build_lm(model_name: str):
    """lm-eval vLLM model for the non-math tasks (thinking off, greedy)."""
    from lm_eval.models.vllm_causallms import VLLM

    gen = config.load("generation")
    mode, eng = gen["other"], gen["engine"]
    extra = {"language_model_only": True} if models.get(model_name).get("language_model_only") else {}
    return VLLM(
        pretrained=str(models.model_dir(model_name)),
        tokenizer=str(models.tokenizer_path(model_name)),
        dtype=eng["dtype"],
        max_model_len=mode["max_model_len"],
        max_gen_toks=mode["sampling"]["max_gen_toks"],
        batch_size=eng["chunk_size"],
        seed=eng["seed"],
        gpu_memory_utilization=eng["gpu_memory_utilization"],
        enable_thinking=mode["enable_thinking"],
        **extra,
    )


# --------------------------------------------------------------------------- shard

def _atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def evaluate_units(lm, task: str, lang_list: list[str], run_dir: Path, meta: dict, limit=None):
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager

    tm = TaskManager(include_path=str(config.TASKS_DIR), include_defaults=False)
    for lang in lang_list:
        unit = run_dir / task / lang
        if (unit / "results.json").exists():
            print(f"[done] {task}/{lang}", flush=True)
            continue
        unit.mkdir(parents=True, exist_ok=True)
        name = f"afbgen_{task}_{lang}"
        t0 = time.time()
        print(f"[run ] {name}", flush=True)
        out = simple_evaluate(
            model=lm,
            tasks=[name],
            task_manager=tm,
            apply_chat_template=True,
            system_instruction=None,          # note 4.3: no system message
            log_samples=True,
            use_cache=str(unit / "cache"),
            bootstrap_iters=0,
            limit=limit,
        )
        samples = out["samples"][name]
        _atomic_write(unit / "samples.jsonl",
                      "".join(json.dumps(s, ensure_ascii=False, default=str) + "\n" for s in samples))
        res = {
            "task": task, "lang": lang, "lm_eval_task": name, "n": len({s["doc_id"] for s in samples}),
            "metrics": {k: v for k, v in out["results"][name].items() if k != "alias"},
            "seconds": round(time.time() - t0, 1), "limit": limit,
            **meta,
        }
        _atomic_write(unit / "results.json", json.dumps(res, indent=1, ensure_ascii=False, default=str))
        print(f"[ok  ] {name} {res['metrics']} ({res['seconds']}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--task", required=True, choices=langs.TASKS + langs.EXTRA_TASKS)
    ap.add_argument("--langs", required=True,
                    help="comma-separated, e.g. eng,hau  (flores: hau_Latn-eng_Latn; afrimgsm: hau:0/2,hau:1/2)")
    ap.add_argument("--limit", type=int, default=None, help="debug only: first N docs per language")
    ap.add_argument("--model", default=models.DEFAULT, help="name in configs/models.yaml")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-any-gpu", action="store_true", help="debug only")
    a = ap.parse_args()

    lang_list = a.langs.split(",")
    bad = [l for l in lang_list if l.split(":")[0] not in langs.TASK_LANGS[a.task]]
    if bad:
        sys.exit(f"unknown languages for {a.task}: {bad}")
    run_dir = a.run_dir if a.limit is None else a.run_dir.with_name(a.run_dir.name + f"-limit{a.limit}")

    prov = provenance(a.allow_dirty)
    models.get(a.model)
    lock_run_manifest(run_dir, prov, a.model)
    check_model(a.model)
    gpu = check_gpu(a.allow_any_gpu)
    meta = {"gpu": gpu, "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "infra_version": INFRA_VERSION, "git": prov["git"], "packages": prov["packages"],
            "model": {"name": a.model, **models.get(a.model)}}
    print(json.dumps(meta, indent=1), flush=True)
    if a.task == "afrimgsm":
        from afbgen import math_runner

        llm = math_runner.build_llm(a.model)
        for spec in lang_list:
            lang, p, k = math_runner.parse_unit(spec)
            math_runner.run_unit(llm, lang, p, k, run_dir, meta, limit=a.limit)
        return
    lm = build_lm(a.model)
    evaluate_units(lm, a.task, lang_list, run_dir, meta, limit=a.limit)


if __name__ == "__main__":
    main()
