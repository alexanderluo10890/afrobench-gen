"""Submit a full evaluation as parallel L40S jobs (one per benchmark x language group).

    source slurm/env.sh
    python slurm/submit.py --run v1.0.0-afriqueqwen-v2            # dry run: print sbatch lines
    python slurm/submit.py --run v1.0.0-afriqueqwen-v2 --submit   # submit
    python slurm/submit.py --run ... --submit --tasks flores,sib   # subset

Idempotent: languages that already have results.json are left out, shards with nothing left
are not submitted, so rerunning after failures/timeouts only resubmits missing work.
A SSA-COMET job is chained after the FLORES jobs (afterany).
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from afbgen import langs  # noqa: E402


def queued_units() -> set[tuple[str, str, str]]:
    """(run, task, unit) of my pending or running jobs, read from the Slurm comment that
    submit.py sets. Resubmitting one of these would start a second writer on the same files."""
    out = subprocess.run(["squeue", "--me", "-h", "-o", "%k"], capture_output=True, text=True).stdout
    busy = set()
    for line in out.splitlines():
        if line.startswith("afb|"):
            _, run, task, units = line.strip().split("|", 3)
            busy |= {(run, task, u) for u in units.split(",")}
    return busy


def shards(task: str, cfg: dict, run_dir: Path, smoke: bool, busy: set) -> list[list[str]]:
    todo = [l for l in langs.TASK_LANGS[task] if not (run_dir / task / l / "results.json").exists()]
    if task == "afrimgsm":
        # one job per (language, question range); a smoke test runs everything in one job
        k = 1 if smoke else cfg["parts_per_language"]
        units = [f"{l}:{p}/{k}" for l in todo for p in range(k)]
        units = [u for u in units if (run_dir.name, task, u) not in busy]
        return ([units] if units else []) if smoke else [[u] for u in units]
    todo = [l for l in todo if (run_dir.name, task, l) not in busy]
    per_job = 99 if smoke else cfg["languages_per_job"]
    return [todo[i:i + per_job] for i in range(0, len(todo), per_job)]


def sbatch(args: list[str], submit: bool) -> str | None:
    print(" ".join(args))
    if not submit:
        return None
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    return re.search(r"(\d+)", out).group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name; results go to $AFB_RESULTS/<run>")
    ap.add_argument("--tasks", default=",".join(langs.TASKS))
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--limit", type=int, help="smoke test: first N docs per language, results in <run>-limitN")
    ap.add_argument("--model", default="afriqueqwen-v2", help="name in configs/models.yaml")
    ap.add_argument("--math-parts", type=int, help="AfriMGSM jobs per language (default: configs/shards.yaml)")
    a = ap.parse_args()

    if "AFB_RESULTS" not in os.environ:
        sys.exit("source slurm/env.sh first")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout
    if dirty.strip():
        sys.exit("uncommitted changes in the repo; jobs would refuse to run. Commit first.")

    run_dir = Path(os.environ["AFB_RESULTS"]) / (a.run if a.limit is None else f"{a.run}-limit{a.limit}")
    if a.limit is not None:
        os.environ["AFB_LIMIT"] = str(a.limit)  # sbatch exports the environment to the job
    os.environ["AFB_MODEL"] = a.model             # read by eval.sbatch
    logs = Path(os.environ["AFB_ROOT"]) / "logs" / run_dir.name
    logs.mkdir(parents=True, exist_ok=True)
    cluster_args = os.environ["AFB_SBATCH_ARGS"].split()
    cfg = yaml.safe_load((REPO / "configs" / "shards.yaml").read_text())
    if a.math_parts:
        cfg["afrimgsm"]["parts_per_language"] = a.math_parts
    if a.model not in yaml.safe_load((REPO / "configs" / "models.yaml").read_text()):
        sys.exit(f"unknown model {a.model}")

    flores_ids, n = [], 0
    busy = queued_units()
    if busy:
        print(f"skipping {sum(r == run_dir.name for r, _, _ in busy)} units of this run that are already queued or running")
    for task in a.tasks.split(","):
        for i, group in enumerate(shards(task, cfg[task], run_dir, a.limit is not None, busy)):
            name = f"afb-{task}-{i}"
            jid = sbatch(["sbatch", *cluster_args, f"--job-name={name}", f"--output={logs}/{name}.out",
                          f"--comment=afb|{run_dir.name}|{task}|{','.join(group)}",
                          str(REPO / "slurm" / "eval.sbatch"), str(run_dir if a.limit is None else Path(os.environ["AFB_RESULTS"]) / a.run), task, ",".join(group), str(REPO)],
                         a.submit)
            n += 1
            if task == "flores" and jid:
                flores_ids.append(jid)

    if "flores" in a.tasks.split(","):
        dep = [f"--dependency=afterany:{':'.join(flores_ids)}"] if flores_ids else []
        sbatch(["sbatch", *cluster_args, *dep, "--job-name=afb-comet", f"--output={logs}/afb-comet.out",
                str(REPO / "slurm" / "comet.sbatch"), str(run_dir), str(REPO)], a.submit)
        n += 1
    print(f"\n{n} jobs {'submitted' if a.submit else '(dry run; add --submit)'}; logs in {logs}")


if __name__ == "__main__":
    main()
