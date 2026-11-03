"""Frozen settings: they must match the generator and the card authors' note (2026-10-29)."""
import subprocess
import sys
from pathlib import Path

from afbgen import config, langs

ROOT = Path(__file__).resolve().parents[1]


def test_task_files_match_generator():
    r = subprocess.run([sys.executable, "scripts/gen_tasks.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_generation_matches_note():
    g = config.load("generation")
    m, o = g["math"], g["other"]
    assert (m["enable_thinking"], o["enable_thinking"]) == (True, False)
    assert (m["max_model_len"], o["max_model_len"]) == (12288, 8192)
    assert m["n_samples"] == 16
    assert m["sampling"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                             "presence_penalty": 1.5, "repetition_penalty": 1.0, "max_tokens": 10240, "seed": 42}
    assert o["sampling"] == {"temperature": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
                             "max_gen_toks": 1024}
    assert g["engine"]["language_model_only"] is True


def test_math_instructions():
    ins = config.load("math_instructions")
    assert set(ins) == {langs.TO_FLORES[l] for l in langs.TASK_LANGS["afrimgsm"]}
    assert ins["eng_Latn"] == ("You are a helpful assistant that solves math word problems.\n"
                               "Please reason step by step in English, and put your final answer within \\boxed{}.")
    assert ins["swh_Latn"].startswith("Wewe ni msaidizi mwenye manufaa")
    assert all(v.count("\n") == 1 and "\\boxed{}" in v for v in ins.values())


def test_language_lists():
    n = {t: len(v) for t, v in langs.TASK_LANGS.items()}
    assert n == {"afrimgsm": 18, "afrimmlu": 18, "afrixnli": 18, "injongo": 17, "belebele": 25, "sib": 26,
                 "flores": 48, "globalpiqa": 136, "mmluprox": 29, "mhumaneval": 55}
    assert all(c in langs.FLORES_NAMES for c in langs.FLORES_LANGS)


def test_pins():
    p = config.load("pins")
    c = p["comet"]
    assert c["primary"] == "mtl-final"
    assert c["variants"]["mtl-final"] == {"repo": "McGill-NLP/ssa-comet-mtl-final",
                                          "revision": "4df7a3d2eda822a3c5b79e7a345261d5166207ce",
                                          "output": "comet_mtl_final.json",
                                          # the encoder named inside model.ckpt (not hparams.yaml)
                                          "encoder_repo": "Davlan/afro-xlmr-large-114L",
                                          "encoder_revision": "3f5e27bb99b577ee1f4ada598fdc4d3497b65c6c"}
    assert p["datasets"]["flores"]["repo"] == "openlanguagedata/flores_plus"
    assert p["datasets"]["flores"]["revision"] == "5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06"
