"""Diagnostics for a finished run, read from the saved samples (no GPU needed).

    python -m afbgen.diagnose --run-dir $AFB_RESULTS/v1.0.0                 # per-language table
    python -m afbgen.diagnose --run-dir ... --examples belebele hau 5       # 5 wrong answers

Per benchmark and language it reports what can explain a score:
  MC (AfriMMLU, AfriXNLI, Belebele, SIB): predicted vs gold letter distribution, unparsed answers
  Injongo: unparsed answers, most common wrong label
  AfriMGSM: answers without \\boxed{}, very long outputs (thinking likely hit the token limit)
  FLORES: SSA-COMET, multi-line / prefixed / empty outputs, length ratio vs reference
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from afbgen import langs

PRIMARY_FILTER = {"afrimgsm": "boxed_or_last", "flores": "clean"}  # everything else: strict
LONG_CHARS = 20000  # ~ >5k tokens of output; with a 10,240-token budget this flags runaway thinking


def load(run_dir: Path, task: str, lang: str) -> list[dict]:
    want = PRIMARY_FILTER.get(task, "strict")
    out = []
    for line in open(run_dir / task / lang / "samples.jsonl", encoding="utf-8"):
        s = json.loads(line)
        if s.get("filter") != want:
            continue
        raw = s["resps"][0][0] if isinstance(s["resps"][0], list) else s["resps"][0]
        pred = s["filtered_resps"][0] if isinstance(s["filtered_resps"], list) else s["filtered_resps"]
        acc = s.get("acc", s.get("metrics", {}).get("acc") if isinstance(s.get("metrics"), dict) else None)
        out.append({"id": s["doc_id"], "doc": s["doc"], "raw": raw or "", "pred": pred or "",
                    "target": s["doc"]["target"], "acc": acc})
    return sorted(out, key=lambda x: x["id"])


def pct(n, d):
    return f"{100 * n / d:5.1f}" if d else "    -"


def dist(letters, n_opt):
    c = Counter(letters)
    return " ".join(f"{L}{100 * c[L] / len(letters):.0f}" for L in "ABCDEFG"[:n_opt])


def report(run_dir: Path):
    for task in langs.TASKS:
        print(f"\n=== {task}")
        if task in ("afrimmlu", "afrixnli", "belebele", "sib"):
            print("lang   acc  unparsed  predicted%                 gold%")
        elif task == "injongo":
            print("lang   acc  unparsed  most common wrong prediction")
        elif task == "afrimgsm":
            print("lang   acc  no-number  no-boxed  long-output  median-chars")
        else:
            print("dir      comet  chrF++  multiline  prefixed  empty  len-ratio")
        for lang in langs.TASK_LANGS[task]:
            unit = run_dir / task / lang
            if not (unit / "samples.jsonl").exists():
                print(f"{lang:6s} missing")
                continue
            S = load(run_dir, task, lang)
            n = len(S)
            acc = 100 * mean(s["pred"] == s["target"] if task != "afrimgsm" else bool(s["acc"]) for s in S)
            if task in ("afrimmlu", "afrixnli", "belebele", "sib"):
                k = len(S[0]["doc"]["options"])
                print(f"{lang:6s}{acc:5.1f}  {pct(sum(not s['pred'] for s in S), n)}     "
                      f"{dist([s['pred'] for s in S if s['pred']], k):26s} {dist([s['target'] for s in S], k)}")
            elif task == "injongo":
                wrong = Counter(s["pred"] or "<unparsed>" for s in S if s["pred"] != s["target"]).most_common(1)
                print(f"{lang:6s}{acc:5.1f}  {pct(sum(not s['pred'] for s in S), n)}     {wrong[0] if wrong else ''}")
            elif task == "afrimgsm":
                L = [len(s["raw"]) for s in S]
                print(f"{lang:6s}{acc:5.1f}  {pct(sum(not s['pred'] for s in S), n)}      "
                      f"{pct(sum('boxed{' not in s['raw'] for s in S), n)}     "
                      f"{pct(sum(x > LONG_CHARS for x in L), n)}       {median(L):8.0f}")
            else:
                comet = json.loads((unit / "comet.json").read_text())["ssa_comet"] if (unit / "comet.json").exists() else float("nan")
                res = json.loads((unit / "results.json").read_text())["metrics"]
                chrf = next((v for k, v in res.items() if k.startswith("chrfpp")), float("nan"))
                tgt = S[0]["doc"]["tgt_name"]
                ratio = mean(len(s["pred"]) / max(1, len(s["target"])) for s in S)
                print(f"{lang:8s}{comet:6.1f}  {chrf:6.1f}  "
                      f"{pct(sum(len([l for l in s['raw'].splitlines() if l.strip()]) > 1 for s in S), n)}     "
                      f"{pct(sum(s['raw'].strip().lower().startswith(tgt.lower() + ':') for s in S), n)}   "
                      f"{pct(sum(not s['pred'] for s in S), n)}   {ratio:5.2f}")


def examples(run_dir: Path, task: str, lang: str, k: int):
    S = [s for s in load(run_dir, task, lang) if s["pred"] != s["target"]]
    print(f"{len(S)} wrong in {task}/{lang}; showing {min(k, len(S))}")
    for s in S[:k]:
        d = s["doc"]
        q = d.get("question") or d.get("text") or d.get("source") or ""
        print(f"\n--- doc {s['id']}  target={s['target']!r}  extracted={s['pred']!r}")
        print("Q:", q[-400:].replace("\n", " | "))
        if "options" in d:
            print("options:", " | ".join(f"{'ABCDEFG'[i]}. {o[:60]}" for i, o in enumerate(d["options"])))
        print("RAW:", s["raw"][-600:].replace("\n", " | "))


def show_prompt(run_dir: Path, task: str, lang: str):
    """The exact string sent to vLLM (chat template applied) for the first doc, as logged."""
    s = json.loads(open(run_dir / task / lang / "samples.jsonl", encoding="utf-8").readline())
    args = s["arguments"]
    first = next(iter(args.values())) if isinstance(args, dict) else args[0]
    prompt, gen = (first["arg_0"], first.get("arg_1")) if isinstance(first, dict) else (first[0], first[1])
    print(f"=== {task}/{lang} doc {s['doc_id']}  (shown with repr: \\n = newline)")
    print(repr(prompt[:1500] + (" …" if len(prompt) > 1500 else "")))
    print("roles:", " -> ".join(seg.split("\n", 1)[0] for seg in prompt.split("<|im_start|>")[1:]))
    print("gen kwargs:", gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--examples", nargs=3, metavar=("TASK", "LANG", "N"))
    ap.add_argument("--prompts", action="store_true", help="exact prompt + gen kwargs per benchmark")
    a = ap.parse_args()
    if a.prompts:
        for task, lang in [("afrimgsm", "hau"), ("afrimgsm", "swa"), ("afrimmlu", "hau"), ("afrixnli", "hau"),
                           ("belebele", "hau"), ("injongo", "hau"), ("sib", "hau"), ("flores", "hau-eng")]:
            show_prompt(a.run_dir, task, lang)
    elif a.examples:
        examples(a.run_dir, a.examples[0], a.examples[1], int(a.examples[2]))
    else:
        report(a.run_dir)


if __name__ == "__main__":
    main()
