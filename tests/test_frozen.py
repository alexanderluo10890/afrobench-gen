"""The task YAMLs and settings are frozen: they must match the generator and the card."""
import subprocess
import sys
from pathlib import Path

from afbgen import config

ROOT = Path(__file__).resolve().parents[1]


def test_task_files_match_generator():
    r = subprocess.run([sys.executable, "scripts/gen_tasks.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_generation_matches_model_card():
    g = config.load("generation")
    m, o = g["math"], g["other"]
    assert m["enable_thinking"] is True and o["enable_thinking"] is False
    assert (m["max_model_len"], o["max_model_len"]) == (12288, 8192)
    assert m["sampling"] | {} == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                                  "presence_penalty": 1.5, "repetition_penalty": 1.0,
                                  "max_gen_toks": 10240, "seed": 1234}
    assert o["sampling"] == {"temperature": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
                             "max_gen_toks": 1024}
    # `do_sample` must stay out of the task configs: lm-eval's cache skips requests that
    # set do_sample=True, which would break resume-after-preemption for AfriMGSM.
    assert "do_sample" not in m["sampling"]
    assert g["engine"]["language_model_only"] is True


def test_math_prompts_from_card():
    assert config.math_system_prompt("eng").endswith(r"put your final answer within \boxed{}.")
    assert config.math_system_prompt("swa").startswith("Wewe ni msaidizi")
    assert "step by step in Hausa" in config.math_system_prompt("hau")


def test_reported_flores_metric_is_stl():
    c = config.load("pins")["comet"]
    assert c["primary"] == "stl"
    assert c["variants"]["stl"] == {"repo": "McGill-NLP/ssa-comet-stl",
                                    "revision": "cd62867ed8fe47527060e6e36662f786d6b74cf7",
                                    "output": "comet_stl.json"}
    assert c["variants"]["mtl"]["output"] == "comet.json"
