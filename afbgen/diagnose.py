"""Diagnostics for a run, read from the saved outputs (no GPU needed).

    python -m afbgen.diagnose --run-dir $AFB_RESULTS/<run>                 # per-unit table
    python -m afbgen.diagnose --run-dir ... --examples belebele hau_Latn 5  # 5 wrong answers
    python -m afbgen.diagnose --run-dir ... --prompts                       # exact logged prompts

MC/Injongo: predicted vs gold distribution, unparsed replies. AfriMGSM: accuracy, truncated
samples (hit max_tokens), samples without \\boxed{}, median output tokens. FLORES: SSA-COMET,
empty cleaned replies, length ratio.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from afbgen import config, langs

COMET_FILE = config.load("pins")["comet"]["variants"][config.load("pins")["comet"]["primary"]]["output"]


def lm_samples(run_dir: Path, task: str, unit: str, flt: str) -> list[dict]:
    out = []
    for line in open(run_dir / task / unit / "samples.jsonl", encoding="utf-8"):
        s = json.loads(line)
        if s.get("filter") != flt:
            continue
        raw = s["resps"][0][0] if isinstance(s["resps"][0], list) else s["resps"][0]
        pred = s["filtered_resps"][0] if isinstance(s["filtered_resps"], list) else s["filtered_resps"]
        out.append({"id": s["doc_id"], "doc": s["doc"], "raw": raw or "", "pred": pred or "",
                    "target": s["doc"]["target"], "args": s.get("arguments")})
    return sorted(out, key=lambda x: x["id"])


def math_records(run_dir: Path, lang: str) -> list[dict]:
    recs = {}
    for p in sorted((run_dir / "afrimgsm" / lang).glob("part*.jsonl")):
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
                recs[r["doc_id"]] = r
            except json.JSONDecodeError:
                pass
    return [recs[k] for k in sorted(recs)]


def pct(n, d):
    return f"{100 * n / d:5.1f}" if d else "    -"


def report(run_dir: Path):
    for task in langs.TASKS:
        print(f"\n=== {task}")
        if task == "afrimgsm":
            print("lang   questions   acc  truncated  no-boxed  median-tokens")
            for lang in langs.TASK_LANGS[task]:
                R = math_records(run_dir, lang)
                if not R:
                    print(f"{lang:6s} missing")
                    continue
                c = [x for r in R for x in r["correct"]]
                fr = [x for r in R for x in r["finish_reasons"]]
                nb = [("boxed{" not in t) for r in R for t in r["outputs"]]
                tk = [x for r in R for x in r["num_tokens"]]
                print(f"{lang:6s} {len(R):9d}  {100 * mean(c):5.1f}  {pct(fr.count('length'), len(fr))}     "
                      f"{pct(sum(nb), len(nb))}   {median(tk):8.0f}")
            continue
        if task == "flores":
            print("direction                comet  skipped  len-ratio")
        else:
            print("unit        acc  unparsed  predicted%  (gold%)")
        for unit in langs.TASK_LANGS[task]:
            if not (run_dir / task / unit / "samples.jsonl").exists():
                print(f"{unit:12s} missing")
                continue
            if task == "flores":
                S = lm_samples(run_dir, task, unit, "clean")
                cf = run_dir / task / unit / COMET_FILE
                c = json.loads(cf.read_text()) if cf.exists() else {}
                ratio = mean(len(s["pred"]) / max(1, len(s["target"])) for s in S)
                print(f"{unit:22s} {c.get('ssa_comet', float('nan')):6.1f}  {c.get('skipped_empty', '-'):>7}  {ratio:6.2f}")
                continue
            S = lm_samples(run_dir, task, unit, "last_option")
            acc = 100 * mean(s["pred"].lower() == s["target"].lower() for s in S)
            pd_, gd = Counter(s["pred"] for s in S), Counter(s["target"] for s in S)
            keys = sorted(gd, key=lambda k: (len(k), k))[:7]
            dist = " ".join(f"{k}{100 * pd_[k] / len(S):.0f}({100 * gd[k] / len(S):.0f})" for k in keys)
            print(f"{unit:10s} {acc:5.1f}  {pct(pd_['[invalid]'], len(S))}     {dist}")


def show_prompts(run_dir: Path):
    for task in langs.TASKS:
        unit = langs.TASK_LANGS[task][4]
        if task == "afrimgsm":
            R = math_records(run_dir, unit)
            if R:
                print(f"=== afrimgsm/{unit}\n{R[0]['prompt']!r}\n")
            continue
        f = run_dir / task / unit / "samples.jsonl"
        if f.exists():
            s = json.loads(f.open(encoding="utf-8").readline())
            args = s["arguments"]
            first = next(iter(args.values())) if isinstance(args, dict) else args[0]
            prompt, gen = (first["arg_0"], first.get("arg_1")) if isinstance(first, dict) else (first[0], first[1])
            print(f"=== {task}/{unit}\n{prompt[:1800]!r}\ngen kwargs: {gen}\n")


def examples(run_dir: Path, task: str, unit: str, k: int):
    if task == "afrimgsm":
        R = [r for r in math_records(run_dir, unit) if not r["correct"][0]]
        for r in R[:k]:
            print(f"\n--- doc {r['doc_id']} target={r['target']} pred={r['preds'][0]} finish={r['finish_reasons'][0]}")
            print("OUTPUT (end):", r["outputs"][0][-700:].replace("\n", " | "))
        return
    flt = "clean" if task == "flores" else "last_option"
    S = [s for s in lm_samples(run_dir, task, unit, flt) if s["pred"].lower() != s["target"].lower()]
    print(f"{len(S)} wrong in {task}/{unit}; showing {min(k, len(S))}")
    for s in S[:k]:
        print(f"\n--- doc {s['id']} target={s['target']!r} extracted={s['pred']!r}")
        print("RAW:", s["raw"][-500:].replace("\n", " | "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--examples", nargs=3, metavar=("TASK", "UNIT", "N"))
    ap.add_argument("--prompts", action="store_true")
    a = ap.parse_args()
    if a.prompts:
        show_prompts(a.run_dir)
    elif a.examples:
        examples(a.run_dir, a.examples[0], a.examples[1], int(a.examples[2]))
    else:
        report(a.run_dir)


if __name__ == "__main__":
    main()
