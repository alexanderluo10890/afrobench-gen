"""Score finished FLORES directions with SSA-COMET (card authors' note, 4.4).

    python -m afbgen.comet_score --run-dir $AFB_RESULTS/<run>                  # reported: mtl-final
    python -m afbgen.comet_score --run-dir $AFB_RESULTS/<run> --variant mtl    # diagnostic

src = source sentence, mt = cleaned reply (tasklib.clean_translation), ref = reference.
Segments with an empty source, reference or cleaned reply are skipped. Direction score =
mean segment score x 100. Each direction's file is written atomically and skipped on rerun.
"""

import argparse
import json
import os
from pathlib import Path
from statistics import mean

from afbgen import config, langs
from afbgen.run_eval import _atomic_write, check_gpu


def triples(unit: Path) -> list[dict]:
    out = []
    for line in open(unit / "samples.jsonl", encoding="utf-8"):
        s = json.loads(line)
        if s.get("filter") != "clean":
            continue
        mt = s["filtered_resps"]
        mt = mt[0] if isinstance(mt, list) else mt
        out.append({"src": s["doc"]["source"], "mt": mt or "", "ref": s["doc"]["target"], "doc_id": s["doc_id"]})
    return sorted(out, key=lambda x: x["doc_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--allow-any-gpu", action="store_true")
    ap.add_argument("--variant", default=None, help="default: configs/pins.yaml comet.primary (mtl-final)")
    a = ap.parse_args()
    check_gpu(a.allow_any_gpu)
    comet_cfg = config.load("pins")["comet"]
    a.variant = a.variant or comet_cfg["primary"]
    pins = comet_cfg["variants"][a.variant]
    out_name = pins["output"]
    ckpt = Path(os.environ["AFB_DATA_DIR"]) / "models" / f"ssa-comet-{a.variant}"
    if (ckpt / ".afb_revision").read_text().strip() != pins["revision"]:
        raise SystemExit(f"{ckpt} is not the pinned revision; run `python -m afbgen.prepare --only comet`")

    from comet import load_from_checkpoint

    model = None
    units = sorted(p.parent for p in (a.run_dir / "flores").glob("*/results.json"))
    for unit in units:
        if (unit / out_name).exists():
            continue
        if model is None:
            model = load_from_checkpoint(str(ckpt / "checkpoints" / "model.ckpt"))
        data = triples(unit)
        kept = [d for d in data if d["src"].strip() and d["ref"].strip() and d["mt"].strip()]
        pred = model.predict([{k: d[k] for k in ("src", "mt", "ref")} for d in kept],
                             batch_size=a.batch_size, gpus=1, progress_bar=False)
        seg = [float(x) for x in pred.scores]
        res = {"ssa_comet": 100 * mean(seg), "n": len(kept), "skipped_empty": len(data) - len(kept),
               "model": pins["repo"], "revision": pins["revision"],
               "segment_scores": dict(zip([d["doc_id"] for d in kept], seg))}
        _atomic_write(unit / out_name, json.dumps(res))
        print(f"[ok  ] {unit.name} ssa_comet={res['ssa_comet']:.2f} (skipped {res['skipped_empty']})", flush=True)

    sc = {u.name: json.loads((u / out_name).read_text())["ssa_comet"] for u in units if (u / out_name).exists()}
    print(f"{len(sc)}/{len(langs.TASK_LANGS['flores'])} directions scored with {pins['repo']}")
    if len(sc) == len(langs.TASK_LANGS["flores"]):
        print(f"FLORES (48 directions, SSA-COMET-{a.variant}): {mean(sc.values()):.2f}  (card: 62.29)", flush=True)


if __name__ == "__main__":
    main()
