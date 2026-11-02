"""AfriMGSM exactly as in the card run (card authors' note, 4.2), preemption-safe.

    python -m afbgen.run_eval --run-dir R --task afrimgsm --langs hau:0/2,hau:1/2

One user message (instruction + blank line + question, no system message), chat template
with enable_thinking=True, vLLM n=16 samples per question with seed=42, answer extracted from
the FULL output (thinking included). Language score = mean accuracy over 250 x 16 samples.

A unit "lang:p/K" covers questions [p*n//K, (p+1)*n//K). Each finished question is appended to
R/afrimgsm/<lang>/part<p>of<K>.jsonl (flush + fsync after every vLLM chunk); a restarted job
skips the questions already there. When all K parts of a language are complete, results.json
is written (atomically) next to them.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from statistics import mean

from afbgen import config, tasklib


def parse_unit(spec: str) -> tuple[str, int, int]:
    lang, _, part = spec.partition(":")
    p, k = (part or "0/1").split("/")
    return lang, int(p), int(k)


def _part_file(run_dir: Path, lang: str, p: int, k: int) -> Path:
    return run_dir / "afrimgsm" / lang / f"part{p}of{k}.jsonl"


def _load_done(path: Path) -> dict[int, dict]:
    """Records already on disk; drops a partially written last line (killed mid-write)."""
    if not path.exists():
        return {}
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raw = raw[: raw.rfind(b"\n") + 1]
        path.write_bytes(raw)
    done = {}
    for line in raw.decode("utf-8").splitlines():
        try:
            r = json.loads(line)
            done[r["doc_id"]] = r
        except (json.JSONDecodeError, KeyError):
            pass
    return done


def build_llm(model_name: str):
    from vllm import LLM

    from afbgen import models

    gen = config.load("generation")
    eng = gen["engine"]
    extra = {"language_model_only": True} if models.get(model_name).get("language_model_only") else {}
    return LLM(model=str(models.model_dir(model_name)), tokenizer=str(models.tokenizer_path(model_name)),
               dtype=eng["dtype"], max_model_len=gen["math"]["max_model_len"],
               gpu_memory_utilization=eng["gpu_memory_utilization"], seed=eng["seed"], **extra)


def run_unit(llm, lang: str, p: int, k: int, run_dir: Path, meta: dict, limit: int | None = None):
    from vllm import SamplingParams

    m = config.load("generation")["math"]
    docs = tasklib.load_rows("afrimgsm", lang)[: limit or None]
    n = len(docs)
    lo, hi = p * n // k, (p + 1) * n // k
    path = _part_file(run_dir, lang, p, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(path)
    todo = [i for i in range(lo, hi) if i not in done]
    print(f"[run ] afrimgsm/{lang} part {p}/{k}: questions {lo}-{hi - 1}, {len(todo)} to do", flush=True)

    tok = llm.get_tokenizer()
    instruction = config.math_instruction(lang)
    params = SamplingParams(n=m["n_samples"], **m["sampling"])
    t0 = time.time()
    for start in range(0, len(todo), m["questions_per_chunk"]):
        ids = todo[start:start + m["questions_per_chunk"]]
        prompts = [tok.apply_chat_template([{"role": "user", "content": tasklib.text_math(instruction, docs[i]["question"])}],
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=m["enable_thinking"]) for i in ids]
        outs = llm.generate(prompts, params, use_tqdm=False)
        with open(path, "a", encoding="utf-8") as f:
            for i, prompt, o in zip(ids, prompts, outs):
                texts = [s.text for s in o.outputs]
                rec = {"doc_id": i, "target": docs[i]["target"], "prompt": prompt,
                       "outputs": texts,
                       "finish_reasons": [s.finish_reason for s in o.outputs],
                       "num_tokens": [len(s.token_ids) for s in o.outputs],
                       "preds": [tasklib.extract_math(t) for t in texts],
                       "correct": [tasklib.math_correct(t, docs[i]["target"]) for t in texts]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"[chunk] afrimgsm/{lang} part {p}/{k}: {start + len(ids)}/{len(todo)} questions "
              f"({time.time() - t0:.0f}s)", flush=True)
    finalize(run_dir, lang, k, n, meta)


def finalize(run_dir: Path, lang: str, k: int, n: int, meta: dict):
    from afbgen.run_eval import _atomic_write

    recs = {}
    for p in range(k):
        recs.update(_load_done(_part_file(run_dir, lang, p, k)))
    if sorted(recs) != list(range(n)):
        print(f"[wait] afrimgsm/{lang}: {len(recs)}/{n} questions on disk; other parts still running", flush=True)
        return
    flat = [c for r in recs.values() for c in r["correct"]]
    res = {"task": "afrimgsm", "lang": lang, "n": n, "samples": len(flat),
           "metrics": {"acc": mean(flat),                                   # card metric: mean over n x 16
                       "acc_first_sample": mean(r["correct"][0] for r in recs.values()),
                       "truncated": mean(fr == "length" for r in recs.values() for fr in r["finish_reasons"])},
           "instruction": config.math_instruction(lang), **meta}
    _atomic_write(run_dir / "afrimgsm" / lang / "results.json", json.dumps(res, indent=1, ensure_ascii=False))
    print(f"[ok  ] afrimgsm/{lang} acc={100 * res['metrics']['acc']:.2f} over {len(flat)} samples", flush=True)
