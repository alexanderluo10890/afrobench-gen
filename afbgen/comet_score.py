"""Score finished FLORES directions with SSA-COMET-MTL (reference-based), per direction.

    python -m afbgen.comet_score --run-dir $AFB_RESULTS/<run>

Writes <run>/flores/<dir>/comet.json (atomically) and skips directions that already have
one, so it is safe to rerun or requeue.
"""

import argparse
import json
import os
from pathlib import Path

from afbgen import config
from afbgen.run_eval import _atomic_write, check_gpu


def triples(unit: Path) -> list[dict]:
    out = []
    for line in open(unit / "samples.jsonl", encoding="utf-8"):
        s = json.loads(line)
        if s.get("filter") != "clean":
            continue
        mt = s["filtered_resps"]
        mt = mt[0] if isinstance(mt, list) else mt
        out.append({"src": s["doc"]["source"], "mt": mt, "ref": s["doc"]["target"], "doc_id": s["doc_id"]})
    return sorted(out, key=lambda x: x["doc_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--allow-any-gpu", action="store_true")
    ap.add_argument("--variant", choices=["mtl", "stl"], default="mtl",
                    help="mtl = reported metric (comet.json); stl = diagnostic (comet_stl.json)")
    a = ap.parse_args()
    check_gpu(a.allow_any_gpu)
    pins = config.load("pins")["comet"]
    if a.variant != "mtl":
        pins = pins["variants"][a.variant]
    out_name = "comet.json" if a.variant == "mtl" else f"comet_{a.variant}.json"
    ckpt = Path(os.environ["AFB_DATA_DIR"]) / "models" / f"ssa-comet-{a.variant}"
    if (ckpt / ".afb_revision").read_text().strip() != pins["revision"]:
        raise SystemExit("SSA-COMET checkpoint is not the pinned revision; rerun prepare")

    from comet import load_from_checkpoint

    model = None
    units = sorted(p.parent for p in (a.run_dir / "flores").glob("*/results.json"))
    for unit in units:
        if (unit / out_name).exists():
            continue
        if model is None:
            model = load_from_checkpoint(str(ckpt / "checkpoints" / "model.ckpt"))
        data = triples(unit)
        pred = model.predict([{k: d[k] for k in ("src", "mt", "ref")} for d in data],
                             batch_size=a.batch_size, gpus=1, progress_bar=False)
        res = {"ssa_comet": 100 * float(pred.system_score), "n": len(data),
               "model": pins["repo"], "revision": pins["revision"],
               "segment_scores": [float(x) for x in pred.scores]}
        _atomic_write(unit / out_name, json.dumps(res))
        print(f"[ok  ] {unit.name} ssa_comet={res['ssa_comet']:.2f}", flush=True)
    print(f"{len(units)} finished FLORES directions in {a.run_dir}")
    from statistics import mean
    from afbgen import langs
    sc = {u.name: json.loads((u / out_name).read_text())["ssa_comet"] for u in units if (u / out_name).exists()}
    xe = [sc[f"{l}-eng"] for l in langs.AFRICAN if f"{l}-eng" in sc]
    ex = [sc[f"eng-{l}"] for l in langs.AFRICAN if f"eng-{l}" in sc]
    if xe and ex:
        print(f"SSA-COMET-{a.variant.upper()} (13 African): xx->eng {mean(xe):.1f}  eng->xx {mean(ex):.1f}  "
              f"both {mean(xe + ex):.1f}  (card: 62.3)")


if __name__ == "__main__":
    main()
