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


def shards(task: str, per_job: int, run_dir: Path) -> list[list[str]]:
    todo = [l for l in langs.TASK_LANGS[task] if not (run_dir / task / l / "results.json").exists()]
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
    a = ap.parse_args()

    if "AFB_RESULTS" not in os.environ:
        sys.exit("source slurm/env.sh first")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout
    if dirty.strip():
        sys.exit("uncommitted changes in the repo; jobs would refuse to run. Commit first.")

    run_dir = Path(os.environ["AFB_RESULTS"]) / (a.run if a.limit is None else f"{a.run}-limit{a.limit}")
    if a.limit is not None:
        os.environ["AFB_LIMIT"] = str(a.limit)  # sbatch exports the environment to the job
        cfg_override = 99  # one job per benchmark is plenty for a smoke test
    else:
        cfg_override = None
    logs = Path(os.environ["AFB_ROOT"]) / "logs" / a.run
    logs.mkdir(parents=True, exist_ok=True)
    cluster_args = os.environ["AFB_SBATCH_ARGS"].split()
    cfg = yaml.safe_load((REPO / "configs" / "shards.yaml").read_text())

    flores_ids, n = [], 0
    for task in a.tasks.split(","):
        per_job = cfg_override or cfg[task]["languages_per_job"]
        for i, group in enumerate(shards(task, per_job, run_dir)):
            name = f"afb-{task}-{i}"
            jid = sbatch(["sbatch", *cluster_args, f"--job-name={name}", f"--output={logs}/{name}.out",
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
