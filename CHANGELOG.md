# Changelog

Every change that can move a score bumps `INFRA_VERSION` (afbgen/__init__.py) and gets a tag.

## 1.0.0 — 2026-10-26
- Initial frozen version: AfroBench-Lite (7 benchmarks, 14 languages + FLORES 28 directions)
  as generative tasks with the AfriqueQwen3.5-4B-Instruct-v2 card prompts and settings.
- lm-eval-harness d6de816, vLLM backend, pinned HF revisions in configs/pins.yaml.
- Python lock files (requirements/*.lock) are added by the first `slurm/setup.sh` run.
