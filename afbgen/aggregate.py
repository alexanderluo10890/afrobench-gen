"""Collect a run into CSVs (for Google Sheets) and a model-card-style table.

    python -m afbgen.aggregate --run-dir $AFB_RESULTS/<run>

Averages follow the card run (card authors' note, 4.1): each task is the mean over ALL
languages of its list (afbgen.langs.TASK_LANGS, English/French included where listed);
FLORES is the mean of its 48 direction scores (SSA-COMET-mtl-final x 100); Overall is the
mean of the 7 task averages.

Writes into <run>/:
  scores_long.csv   one row per (task, unit, metric)
  scores_wide.csv   primary metric per language (FLORES-200 code), one column per task;
                    FLORES as two columns (xx->eng, eng->xx)
  summary.csv       task averages + Overall
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from afbgen import INFRA_VERSION, config, langs

COMET = config.load("pins")["comet"]
PRIMARY = {"afrimgsm": "acc", "flores": f"ssa_comet_{COMET['primary']}",
           **{t: "exact_match,last_option" for t in ["afrimmlu", "afrixnli", "belebele", "injongo", "sib",
                                                     *langs.EXTRA_TASKS]}}
CARD_COLUMNS = [("afrimgsm", "AfriMGSM"), ("afrimmlu", "AfriMMLU"), ("afrixnli", "AfriXNLI"),
                ("belebele", "Belebele"), ("injongo", "Injongo"), ("sib", "SIB-200"), ("flores", "FLORES")]
CARD = {"AfriMGSM": 50.8, "AfriMMLU": 46.4, "AfriXNLI": 63.6, "Belebele": 66.2, "Injongo": 58.7,
        "SIB-200": 81.2, "FLORES": 62.3, "Overall": 61.3}


def collect(run_dir: Path):
    rows, missing, versions = [], [], set()
    # card tasks always; extra benchmarks only when this run contains them
    tasks = langs.TASKS + [t for t in langs.EXTRA_TASKS if (run_dir / t).exists()]
    for task in tasks:
        for unit_name in langs.TASK_LANGS[task]:
            unit = run_dir / task / unit_name
            rp = unit / "results.json"
            if not rp.exists():
                missing.append(f"{task}/{unit_name}")
                continue
            r = json.loads(rp.read_text(encoding="utf-8"))
            versions.add((r["infra_version"], r["git"]["commit"]))
            metrics = {k: v for k, v in r["metrics"].items()
                       if "stderr" not in k and isinstance(v, (int, float)) and not isinstance(v, bool)}
            metrics = {k: (100 * v if k.startswith(("acc", "exact_match", "truncated")) else v)
                       for k, v in metrics.items()}
            if task == "flores":
                for variant, v in COMET["variants"].items():
                    cp = unit / v["output"]
                    if cp.exists():
                        metrics[f"ssa_comet_{variant}"] = json.loads(cp.read_text())["ssa_comet"]
                    elif variant == COMET["primary"]:
                        missing.append(f"flores/{unit_name} (SSA-COMET-{variant})")
            for k, v in metrics.items():
                rows.append({"task": task, "lang": unit_name, "metric": k, "value": round(v, 2), "n": r["n"],
                             "primary": k == PRIMARY[task], "infra_version": r["infra_version"],
                             "commit": r["git"]["commit"][:10]})
    return rows, missing, versions


def summarize(rows):
    prim = {(r["task"], r["lang"]): r["value"] for r in rows if r["primary"]}
    summary = {}
    for task, col in CARD_COLUMNS:
        vals = [prim[(task, u)] for u in langs.TASK_LANGS[task] if (task, u) in prim]
        expected = len(langs.TASK_LANGS[task])
        summary[col] = (round(mean(vals), 2) if len(vals) == expected else (round(mean(vals), 2) if vals else None),
                        len(vals), expected)
    complete = [summary[c][0] for _, c in CARD_COLUMNS if summary[c][1] == summary[c][2]]
    summary["Overall"] = (round(mean(complete), 2) if len(complete) == len(CARD_COLUMNS) else None,
                          len(complete), len(CARD_COLUMNS))
    return prim, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    a = ap.parse_args()
    if not a.run_dir.exists():
        raise SystemExit(f"{a.run_dir} does not exist yet (no job has started)")
    rows, missing, versions = collect(a.run_dir)
    if len(versions) > 1:
        raise SystemExit(f"run mixes infra versions/commits: {versions}")

    with open(a.run_dir / "scores_long.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "lang", "metric", "value", "n", "primary", "infra_version", "commit"])
        w.writeheader()
        w.writerows(rows)

    prim, summary = summarize(rows)
    # wide table keyed by FLORES-200 code
    cols = [c for t, c in CARD_COLUMNS if t != "flores"] + ["FLORES xx->eng", "FLORES eng->xx"]
    wide = {}
    for (task, unit), v in prim.items():
        if task == "flores":
            src, tgt = unit.split("-")
            code, col = (src, "FLORES xx->eng") if tgt == "eng_Latn" else (tgt, "FLORES eng->xx")
        else:
            code, col = langs.TO_FLORES.get(unit, unit), dict(CARD_COLUMNS)[task]
        wide.setdefault(code, {})[col] = v
    with open(a.run_dir / "scores_wide.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lang", "name"] + cols)
        for code in sorted(wide):
            w.writerow([code, langs.display(code)] + [wide[code].get(c, "") for c in cols])

    with open(a.run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "score", "card_v2", "units_done", "units_expected"])
        for k, (v, n, e) in summary.items():
            w.writerow([k, "" if v is None else v, CARD[k], n, e])

    print(f"{'':10s}" + "".join(f"{k:>10s}" for k in summary))
    print(f"{'ours':10s}" + "".join(f"{'…' if v is None else f'{v:.1f}':>10s}" for v, n, e in summary.values()))
    print(f"{'card v2':10s}" + "".join(f"{CARD[k]:10.1f}" for k in summary))
    print(f"{'done':10s}" + "".join(f"{f'{n}/{e}':>10s}" for v, n, e in summary.values()))
    extra = [t for t in langs.EXTRA_TASKS if any(k[0] == t for k in prim)]
    if extra:
        print("\nextra benchmarks (not on the card; accuracy %):")
        with open(a.run_dir / "summary_extra.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["benchmark", "all_languages", "n_all", "african_languages", "n_african", "units_done",
                        "units_expected"])
            for t in extra:
                vals = {u: prim[(t, u)] for u in langs.TASK_LANGS[t] if (t, u) in prim}
                afr = [vals[u] for u in langs.AFRICAN_SUBSET[t] if u in vals]
                row = [t, round(mean(vals.values()), 2), len(vals), round(mean(afr), 2) if afr else "",
                       len(afr), len(vals), len(langs.TASK_LANGS[t])]
                w.writerow(row)
                print(f"  {t:11s} all {row[1]:6.2f} ({row[2]} langs)   African {row[3]} ({row[4]} langs)"
                      f"   done {row[5]}/{row[6]}")
    if missing:
        print(f"\nmissing ({len(missing)}):", ", ".join(missing[:40]), "…" if len(missing) > 40 else "")
    print(f"\nafrobench-gen {INFRA_VERSION}; FLORES = SSA-COMET-{COMET['primary']}")
    print(f"wrote scores_long.csv, scores_wide.csv, summary.csv to {a.run_dir}")


if __name__ == "__main__":
    main()
