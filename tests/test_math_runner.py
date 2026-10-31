"""AfriMGSM runner logic with a fake vLLM: parts, resume after a crash, 16-sample scoring."""
import json
import os
import sys
import types

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("AFB_DATA_DIR"), reason="AFB_DATA_DIR not set")


class FakeTok:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert [m["role"] for m in messages] == ["user"] and enable_thinking and add_generation_prompt
        return "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n<think>\n"


class FakeLLM:
    calls = 0

    def get_tokenizer(self):
        return FakeTok()

    def generate(self, prompts, params, use_tqdm=False):
        FakeLLM.calls += 1
        assert params.n == 16 and params.seed == 42
        outs = []
        for p in prompts:
            sample = lambda j: types.SimpleNamespace(  # noqa: E731  half right, half wrong
                text=f"<think>...</think> \\boxed{{{'GOLD' if j % 2 == 0 else '-1'}}}", finish_reason="stop",
                token_ids=[0] * 10)
            outs.append(types.SimpleNamespace(outputs=[sample(j) for j in range(params.n)]))
        return outs


@pytest.fixture
def fake_vllm(monkeypatch):
    mod = types.ModuleType("vllm")
    mod.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "vllm", mod)


def test_parts_resume_and_score(tmp_path, fake_vllm, monkeypatch):
    from afbgen import math_runner, tasklib

    docs = tasklib.load_rows("afrimgsm", "hau")[:6]
    # make "GOLD" decode to the right answer for each question
    orig = tasklib.math_correct
    monkeypatch.setattr(tasklib, "math_correct", lambda out, gold: orig(out.replace("GOLD", gold), gold))
    meta = {"infra_version": "test", "git": {"commit": "x"}}

    math_runner.run_unit(FakeLLM(), "hau", 0, 2, tmp_path, meta, limit=6)
    assert not (tmp_path / "afrimgsm/hau/results.json").exists()      # part 1 still missing

    # simulate a crash in part 1 after one question, with a half-written line
    part1 = tmp_path / "afrimgsm/hau/part1of2.jsonl"
    rec = {"doc_id": 3, "target": docs[3]["target"], "prompt": "p", "outputs": ["x"] * 16,
           "finish_reasons": ["length"] * 16, "num_tokens": [1] * 16, "preds": [None] * 16, "correct": [False] * 16}
    part1.write_text(json.dumps(rec) + "\n" + '{"doc_id": 4, "tar', encoding="utf-8")
    math_runner.run_unit(FakeLLM(), "hau", 1, 2, tmp_path, meta, limit=6)

    lines = part1.read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["doc_id"] for l in lines] == [3, 4, 5]          # partial line dropped, 3 kept
    res = json.loads((tmp_path / "afrimgsm/hau/results.json").read_text(encoding="utf-8"))
    assert res["n"] == 6 and res["samples"] == 96
    # 5 questions x 8/16 right + question 3 (from the "crashed" run) 0/16 right
    assert res["metrics"]["acc"] == pytest.approx(5 * 8 / 96)
    first = json.loads(lines[1])
    assert first["prompt"].count("<|im_start|>") == 2 and "system" not in first["prompt"]
    assert "\\boxed{}" in first["prompt"] and "\n\n" in first["prompt"]
