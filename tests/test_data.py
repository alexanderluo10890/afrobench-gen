"""Needs the prepared data: AFB_DATA_DIR=... pytest tests/test_data.py"""
import os

import pytest

from afbgen import langs
from afbgen import tasklib as T

pytestmark = pytest.mark.skipif(not os.environ.get("AFB_DATA_DIR"), reason="AFB_DATA_DIR not set")

SIZES = {"afrimgsm": 250, "afrimmlu": 500, "afrixnli": 600, "belebele": 900, "injongo": 640, "sib": 204,
         "flores": 1012}


@pytest.mark.parametrize("task", langs.TASKS)
def test_sizes_and_targets(task):
    if task == "flores" and not (T.data_dir("flores") / ".afb_revision").exists():
        pytest.skip("FLORES not downloaded (gated)")
    for lang in langs.TASK_LANGS[task]:
        rows = T.load_rows(task, lang)
        expected = 622 if (task, lang) == ("injongo", "eng") else SIZES[task]
        assert len(rows) == expected, (lang, len(rows))
        for r in rows:
            assert all(v not in (None, "") for k, v in r.items() if k != "subject"), r
            if "options" in r:
                assert r["target"] == T.LETTERS[r["answer_idx"]]
            if task == "injongo":
                assert r["target"] in T.INJONGO_LABELS
