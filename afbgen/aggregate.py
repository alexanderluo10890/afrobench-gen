"""Collect a run into CSVs (for Google Sheets) and a model-card-style table.

    python -m afbgen.aggregate --run-dir $AFB_RESULTS/<run>

Writes into <run>/:
  scores_long.csv   one row per (task, language, metric)
  scores_wide.csv   primary metric, rows = language, columns = task
  summary.csv       card-style averages (13 African languages; English and French excluded)
and prints the summary plus every (task, language) that is still missing.
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from afbgen import langs

PRIMARY = {
    "afrimgsm": "acc,boxed_or_last",
    "afrimmlu": "acc,strict", "afrixnli": "acc,strict", "belebele": "acc,strict",
    "injongo": "acc,strict", "sib": "acc,strict",
    "flores": "ssa_comet",
}
CARD_COLUMNS = [("afrimgsm", "AfriMGSM"), ("afrimmlu", "AfriMMLU"), ("afrixnli", "AfriXNLI"),
                ("belebele", "Belebele"), ("injongo", "Injongo"), ("sib", "SIB-200"), ("flores", "FLORES")]


def is_african(task: str, lang: str) -> bool:
    if task == "flores":
        return "fra" not in lang
    return lang in langs.AFRICAN


def collect(run_dir: Path):
    rows, missing, versions = [], [], set()
    for task in langs.TASKS:
        for lang in langs.TASK_LANGS[task]:
            unit = run_dir / task / lang
            rp = unit / "results.json"
            if not rp.exists():
                missing.append(f"{task}/{lang}")
                continue
            r = json.loads(rp.read_text(encoding="utf-8"))
            versions.add((r["infra_version"], r["git"]["commit"]))
            metrics = {k: v for k, v in r["metrics"].items() if "stderr" not in k and isinstance(v, (int, float))}
            # accuracies -> percent, like the card
            metrics = {k: (100 * v if k.startswith("acc") else v) for k, v in metrics.items()}
            if task == "flores":
                cp = unit / "comet.json"
                if cp.exists():
                    metrics["ssa_comet"] = json.loads(cp.read_text())["ssa_comet"]
                else:
                    missing.append(f"flores/{lang} (SSA-COMET)")
            for k, v in metrics.items():
                rows.append({"task": task, "lang": lang, "metric": k, "value": round(v, 2), "n": r["n"],
                             "primary": k == PRIMARY[task], "infra_version": r["infra_version"],
                             "commit": r["git"]["commit"][:10]})
    return rows, missing, versions


def summarize(rows):
    prim = {(r["task"], r["lang"]): r["value"] for r in rows if r["primary"]}
    summary = {}
    for task, col in CARD_COLUMNS:
        vals = [v for (t, l), v in prim.items() if t == task and is_african(t, l)]
        expected = sum(is_african(task, l) for l in langs.TASK_LANGS[task])
        summary[col] = (round(mean(vals), 1) if vals else None, len(vals), expected)
    done = [v[0] for v in summary.values() if v[0] is not None and v[1] == v[2]]
    summary["Overall"] = (round(mean(done), 1) if len(done) == len(CARD_COLUMNS) else None, len(done), len(CARD_COLUMNS))
    return prim, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    a = ap.parse_args()
    rows, missing, versions = collect(a.run_dir)
    if len(versions) > 1:
        raise SystemExit(f"run mixes infra versions/commits: {versions}")

    with open(a.run_dir / "scores_long.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["task"])
        w.writeheader()
        w.writerows(rows)

    prim, summary = summarize(rows)
    all_langs = list(dict.fromkeys(["eng"] + langs.AFRICAN + langs.FLORES_DIRECTIONS))
    with open(a.run_dir / "scores_wide.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lang"] + [c for _, c in CARD_COLUMNS])
        for l in all_langs:
            vals = [prim.get((t, l), "") for t, _ in CARD_COLUMNS]
            if any(v != "" for v in vals):
                w.writerow([l] + vals)

    with open(a.run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "score", "languages_done", "languages_expected"])
        for k, (v, n, e) in summary.items():
            w.writerow([k, "" if v is None else v, n, e])

    print("| " + " | ".join(summary) + " |")
    print("|" + "---|" * len(summary))
    print("| " + " | ".join("…" if v is None else f"{v}" + ("" if n == e else f" ({n}/{e})")
                            for v, n, e in summary.values()) + " |")
    if missing:
        print(f"\nmissing ({len(missing)}):", ", ".join(missing))
    print(f"\nwrote scores_long.csv, scores_wide.csv, summary.csv to {a.run_dir}")


if __name__ == "__main__":
    main()
