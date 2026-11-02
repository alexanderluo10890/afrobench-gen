# Changelog

Every change that can move a score bumps `INFRA_VERSION` (afbgen/__init__.py) and gets a tag.

## 2.2.0 — 2026-11-02 — baselines: several models, tokenizer check
- `configs/models.yaml`: pinned registry of the evaluated models (AfriqueQwen3.5-4B-Instruct-v2
  and the baselines Qwen3.5-4B/9B, Sunflower-Qwen3.5-9B, Gemma-3-4B-it, tiny-aya-earth,
  tiny-aya-l2-thinker) with the card's per-benchmark numbers where the card has them.
  `--model` in prepare / run_eval / submit; every model uses the same benchmark settings.
- Tokenizer check before every run (`afbgen/models.py`): the tokenizer vLLM loads must give
  the same ids as the repo's tokenizer.json on digits and African scripts. tiny-aya under
  transformers v5 fails it (CohereTokenizer splits digits one by one; issue #49176); its
  tokenizer is loaded from tokenizer.json as PreTrainedTokenizerFast (`tokenizer_fix`).
- run.json records the model; a run directory cannot mix models.
- AfriMGSM runner passes token ids tokenized WITHOUT added special tokens: the chat template
  already writes the BOS token for models that have one (Gemma, tiny-aya); a text prompt got a
  second BOS from vLLM. Qwen-family models have no BOS, so AfriqueQwen v2.0.0 results are
  unaffected. lm-eval tasks already avoided a double BOS.
- Per-model context cap (`max_model_len` in models.yaml): tiny-aya-earth has 8192 tokens, so its
  AfriMGSM runs at 8192 with max new tokens = min(10240, 8192 - prompt length).
- Jobs import Python from the node-local disk: `slurm/pack_env.sh` packs each environment into
  one tar file; `slurm/local_env.sh` extracts it to `$SLURM_TMPDIR` at job start (falls back to
  the network copy if the tar is missing or was packed from another lock file). Importing
  transformers opens every file of ~400 model folders; on the shared disk some opens failed on
  compute nodes (FileNotFoundError for files that exist), and 12 of ~360 jobs failed 7 times.
- submit.py skips units that are already queued or running (Slurm job comment
  `afb|<run>|<task>|<units>`), so a resubmission never starts a second writer on the same files.
- No change to prompts, decoding or scoring.

## 2.1.0 — 2026-11-01 — extra benchmarks: Global-PIQA, MMLU-ProX-Lite
- New tasks `globalpiqa` (136 languages x 100) and `mmluprox` (29 x 588), pinned in
  configs/pins.yaml. Not on the card; settings are ours (README "Extra benchmarks"): card MC
  template with an adapted letter list, thinking off, greedy, last-option regex + exact match.
- `aggregate` writes `summary_extra.csv` (all-language and African-language means).
- The 7 card benchmarks are unchanged (generation, prompts, scoring identical to 2.0.0).
- **Result** (run v2.1.0-extra on Mila, generation at commit 1740d68): Global-PIQA 68.0 over 136
  languages, 68.6 over the 21 African ones; MMLU-ProX-Lite 28.4 over 29 languages, 25.2 over
  af/sw/wo/yo/zu. MMLU-ProX answers "A" 44.5% of the time (direct answer, thinking off).

## 2.0.0 — 2026-10-31 — settings of the card run
Every setting now follows the card authors' note "How to match the
AfriqueQwen3.5-4B-Instruct-v2 card numbers" (2026-10-29). The note showed that our 1.x numbers
only looked close to the card because two errors cancelled out: our 13-language averages vs
the card's 17-26 languages per task, and lower scores on 5 tasks from different settings.
- **Languages:** the card lists (17-26 per task, English/French where listed; FLORES 24
  languages x 2 directions); averages over all languages of each list.
- **AfriMGSM:** one user message (translated instruction + blank line + question), no system
  message; 16 samples per question (vLLM n=16, seed 42); answer from the full output,
  thinking included (Unicode minus, digit spaces, last box, box without number = wrong).
  New runner `afbgen/math_runner.py` (lm-eval cannot draw n samples per request).
- **AfriXNLI, Belebele, Injongo, SIB-200:** the card-run prompts. Injongo answers with the
  intent number 1-40; SIB topics are in alphabetical order.
- **Extraction:** the last standalone option (lm-eval `regex` filter) + `exact_match`
  (ignore case/punctuation), instead of "reply starts with the letter".
- **FLORES:** FLORES+ (`openlanguagedata/flores_plus` @ 5fec6c1) instead of FLORES-200;
  cleaning = after the last `</think>`, before the first blank line, no prefix removal;
  **SSA-COMET `ssa-comet-mtl-final`** (@ 4df7a3d), empty segments skipped.
- `prepare` now also downloads new languages for datasets already on disk.
- SSA-COMET encoders are pinned per checkpoint: `ssa-comet-mtl-final` is built on
  `Davlan/afro-xlmr-large-114L` (named inside model.ckpt; its hparams.yaml says 76L).
  COMET jobs requeue on crash.
- **Result** (run v2.0.0 on Mila, generation at commit 948cb7a, FLORES scored at fa23e15):
  AfriMGSM 50.8, AfriMMLU 46.4, AfriXNLI 63.6, Belebele 66.3, Injongo 58.6, SIB-200 81.0,
  FLORES 62.31, **Overall 61.3** - card v2: 50.8 / 46.4 / 63.6 / 66.2 / 58.7 / 81.2 / 62.29 / 61.3.

## 1.1.0 — superseded, do not use
1.1.0 reported FLORES with SSA-COMET-STL because that score landed near the card. According
to the card authors' note, the card uses `ssa-comet-mtl-final`, and STL was close by chance.
Choosing a metric for being closer to a target was a mistake on our part.

Original 1.1.0 entry:
- **Reported FLORES metric changed from SSA-COMET-MTL to SSA-COMET-STL**
  (McGill-NLP/ssa-comet-stl @ cd62867). Rescoring the unchanged v1.0.0 translations gives
  STL 61.4 (xx->eng 61.4, eng->xx 61.4) vs the card's 62.3; MTL gave 56.8. MTL is still
  computed (`comet.json`, column `ssa_comet_mtl`).
- Generation, prompts, decoding and answer extraction are unchanged: 1.1.0 results are the
  v1.0.0 generations rescored, so `infra_version` in results.json stays 1.0.0 and the CSVs
  carry `scoring_version` 1.1.0.
- Result (13 African languages): AfriMGSM 50.2, AfriMMLU 47.2, AfriXNLI 64.3,
  Belebele 60.9, Injongo 67.2, SIB-200 79.9, FLORES 61.4, Overall 61.6 (card v2: 61.3).
- Diagnostics: `python -m afbgen.diagnose` (per-language answer distributions, unparsed
  answers, truncated math reasoning, FLORES formatting, `--examples`, `--prompts` with the
  exact logged prompt and roles). Found no pipeline errors.

## 1.0.0 — 2026-10-26
- Initial frozen version: AfroBench-Lite (7 benchmarks, 14 languages + FLORES 28 directions)
  as generative tasks with the AfriqueQwen3.5-4B-Instruct-v2 card prompts and settings.
- lm-eval-harness d6de816, vLLM backend, pinned HF revisions in configs/pins.yaml.
- Python lock files (requirements/*.lock) from the first install on Mila: vLLM 0.32.0,
  torch 2.13.0+cu130, transformers 5.19.0, triton 3.7.1; COMET env unbabel-comet 2.2.7.
- Cluster fixes before the first full run: CUDA module (cuda/13.1.0 on Mila) for vLLM JIT,
  venv on PATH + PyTorch sampler instead of FlashInfer JIT, precompiled bytecode,
  requeue on crash. None of these change prompts, extraction or scoring.
- Run v1.0.0 (Mila, 2026-10-27) was produced at Mila-local commit d36f035 = 0324852 +
  these lock files; this commit has the same file contents.
  Result (13 African languages): AfriMGSM 50.2, AfriMMLU 47.2, AfriXNLI 64.3,
  Belebele 60.9, Injongo 67.2, SIB-200 79.9, FLORES 56.8, Overall 60.9 (card v2: 61.3).
