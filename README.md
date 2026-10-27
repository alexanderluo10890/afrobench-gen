# afrobench-gen

A fixed evaluation harness for **AfroBench-Lite** that uses **generative** tasks and the
prompts and decoding settings from the
[AfriqueQwen3.5-4B-Instruct-v2 model card](https://huggingface.co/McGill-NLP/AfriqueQwen3.5-4B-Instruct-v2).
It runs on L40S GPUs on the Mila and Vulcan clusters.

Native AfroBench scores AfriMMLU, AfriXNLI, Belebele, SIB-200 and Injongo by log-likelihood
(`output_type: multiple_choice`). Here every task is `generate_until`. The model writes an
answer through its chat template, the answer is extracted, and it is scored by exact match.
This matches how the card reports its numbers.

| Benchmark | AfroBench source | Here | Metric |
|---|---|---|---|
| AfriMGSM | `afrimgsm_cot` (generative) | Card math system prompt, thinking on, sampling | Number in the last `\boxed{}`, else the last number; exact match |
| AfriMMLU | log-likelihood | Card MC template (A–D) | Exact match on the letter |
| AfriXNLI | log-likelihood | Card MC template (A–C) | Exact match on the letter |
| Belebele | log-likelihood | Card MC template (A–D), passage plus question | Exact match on the letter |
| SIB-200 | log-likelihood | Card MC template (A–G) | Exact match on the letter |
| Injongo | log-likelihood | Label output, 40 intents | Exact match on the label |
| FLORES | generative | Card translation template | **SSA-COMET-MTL** (×100), plus chrF++ and BLEU |

Languages: the AfroBench-Lite set, which is English plus amh, hau, ibo, kin, lin, lug, orm,
sna, sot, swa, xho, yor and zul. FLORES runs both directions between English and the 13
African languages plus French (28 directions). **Reported averages cover the 13 African
languages only.** English and French are evaluated but left out of the averages, and
"Overall" is the mean of the 7 benchmark averages, as in the card.

## What is frozen, and how

| Item | Pinned in | Enforced by |
|---|---|---|
| Model, datasets, SSA-COMET and its encoder (exact HF commits) | `configs/pins.yaml` | `afbgen.prepare` downloads exactly these, and `run_eval` refuses a model that is not the pinned revision |
| lm-eval-harness commit | `configs/pins.yaml`, `requirements/eval.in` | Installed from git at that commit |
| Python environment | `requirements/*.lock` | `slurm/setup.sh` runs `uv pip sync` from the lock files |
| Decoding settings | `configs/generation.yaml` | `tests/test_frozen.py` checks them against the card |
| Prompts and task YAMLs | `afbgen/tasklib.py`, `scripts/gen_tasks.py` | The YAMLs are generated, and `tests/test_frozen.py` fails if they drift |
| Code version | git commit plus `INFRA_VERSION` | `run_eval` refuses uncommitted code, and every result records the commit. All shards of a run must come from the same commit (`run.json`), and `aggregate` refuses to mix versions |

Any change that can move a score must bump `INFRA_VERSION`, get a `CHANGELOG.md` entry
and a git tag. That covers prompts, extraction, pins and settings.

## Running

One command on a login node does setup, downloads and submission: `bash slurm/bootstrap.sh v1.0.0`.
Step by step:

```bash
# once per cluster, on a login node
git clone <this repo> && cd afrobench-gen
bash slurm/setup.sh                                # uv venvs from the lock files
source slurm/env.sh
hf auth login                                      # FLORES is gated: accept its terms on HF first
$AFB_VENV/bin/python -m afbgen.prepare             # model, datasets, SSA-COMET at the pinned revisions
$AFB_VENV/bin/python -m afbgen.prepare --only verify

# smoke test: 20 docs per language, one job per benchmark
$AFB_VENV/bin/python slurm/submit.py --run v1.0.0 --limit 20 --submit

# full run: ~24 parallel L40S jobs plus 1 SSA-COMET job
$AFB_VENV/bin/python slurm/submit.py --run v1.0.0 --submit
squeue --me

# rerun the same command at any time; it only resubmits unfinished languages
$AFB_VENV/bin/python slurm/submit.py --run v1.0.0 --submit

# results
$AFB_VENV/bin/python -m afbgen.aggregate --run-dir $AFB_RESULTS/v1.0.0
# -> summary.csv, scores_wide.csv, scores_long.csv (import into Google Sheets)
```

### Cluster rules built in

- **L40S only.** Every job requests `--gres=gpu:l40s:1`, and `run_eval` exits if the GPU is
  not an L40S. A100 and H100 are never used.
- **Mila uses the `long` partition** (`slurm/env.sh`). Vulcan uses account `aip-davlan`.
- **3 h limit per job** (`--time=03:00:00`).
- **Parallel.** Each benchmark is split into jobs of a few languages each
  (`configs/shards.yaml`, ~24 jobs), so all benchmarks run at once.
- **Preemption-safe.** Each generation is committed to a per-language SQLite cache as soon
  as its vLLM chunk finishes (`chunk_size` in `configs/generation.yaml`). A language is
  complete once its `results.json` is written; that file is written atomically and last.
  Ten minutes before the time limit a job requeues itself. A requeued or preempted job
  replays the cache and continues where it stopped.

### Output layout

```
$AFB_RESULTS/<run>/
  run.json                       # commit + config every shard must match
  <task>/<lang>/results.json     # metrics, n, GPU, host, job id, commit, package versions
  <task>/<lang>/samples.jsonl    # prompt, raw response, extracted answer, target per doc
  <task>/<lang>/cache_rank0.db   # generation cache (used for resume)
  flores/<dir>/comet.json        # SSA-COMET system and segment scores
  summary.csv  scores_wide.csv  scores_long.csv
```

## Decisions to confirm (the card does not specify these)

1. **Math system prompts for languages other than English and Swahili.** The card gives the
   English and Swahili prompts and says to translate for other languages. Until official
   translations exist, the other 11 languages use the English prompt with the language
   name filled in ("…step by step in Hausa…"). Translations go in
   `configs/math_system_prompts.yaml` under `overrides`.
2. **Thinking and AfriMGSM extraction.** With `enable_thinking=true` the text up to
   `</think>` is dropped, and the `\boxed{}` or last-number rule is applied to the answer
   that follows. If the model never closes its thinking (it hit the 10,240-token limit), the
   rule is applied to the whole output.
3. **AfriXNLI and SIB-200 as multiple choice.** The card's A–D template is generalized to
   A–C (entailment, neutral, contradiction, in that order) and A–G (the 7 SIB categories,
   in AfroBench order).
4. **Belebele.** The card's single `{question}` slot is filled with
   `passage + blank line + question`.
5. **Injongo** has 40 intents, too many for lettered options. The model is asked to reply
   with the label itself (see `text_injongo`), and the label is scored by exact match.
6. **Answer extraction.** `strict` requires the response to *start* with the letter or
   label; markdown, quotes and a trailing period are tolerated. `flexible` also accepts
   "The answer is X" and responses that repeat the option text. **Primary = strict.** Both
   are stored in `scores_long.csv`.
7. **Zero-shot everywhere, and a seed for sampling.** AfroBench-Lite uses 5-shot for
   Belebele, SIB and FLORES, but the card describes zero-shot chat prompting. AfriMGSM is
   sampled at temperature 0.7 with a fixed per-request seed (1234), so the result does not
   depend on batching or resume order. The card does not mention a seed.
8. **FLORES.** The output is trimmed to the first non-empty line, with an echoed
   `"<Language>:"` prefix removed. SSA-COMET is the reference-based `ssa-comet-mtl`
   checkpoint, and its 0–1 scores are multiplied by 100.
9. **Language set and averages.** AfroBench-Lite languages; averages exclude English and
   French (see above).

## Repository layout

```
afbgen/tasklib.py        loaders, prompts, answer extraction, scoring (all task logic)
afbgen/run_eval.py       one shard: load vLLM once, loop languages, resumable
afbgen/comet_score.py    SSA-COMET for finished FLORES directions
afbgen/aggregate.py      CSVs + card-style table
afbgen/prepare.py        pinned downloads
scripts/gen_tasks.py     generates tasks/afrobench_gen/*.yaml
configs/                 pins, generation settings, math prompts, sharding
slurm/                   env.sh, setup.sh, eval.sbatch, comet.sbatch, submit.py
tests/                   extraction, freeze checks, data checks
```
