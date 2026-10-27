# Changelog

Every change that can move a score bumps `INFRA_VERSION` (afbgen/__init__.py) and gets a tag.

## 1.1.0 — 2026-10-27 — FLORES metric: SSA-COMET-STL
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
