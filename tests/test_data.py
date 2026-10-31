"""Needs the prepared data: AFB_DATA_DIR=... pytest tests/test_data.py"""
import os

import pytest

from afbgen import langs
from afbgen import tasklib as T

pytestmark = pytest.mark.skipif(not os.environ.get("AFB_DATA_DIR"), reason="AFB_DATA_DIR not set")

# items per language (card authors' note, 4.3 / 4.4)
SIZES = {"afrimgsm": 250, "afrimmlu": 500, "afrixnli": 600, "belebele": 900, "injongo": 640, "sib": 204,
         "flores": 1012}
EXCEPTIONS = {("injongo", "eng"): 622, ("injongo", "wol"): 638}


@pytest.mark.parametrize("task", langs.TASKS)
def test_sizes_and_targets(task):
    if task == "flores" and not (T.data_dir("flores") / ".afb_revision").exists():
        pytest.skip("FLORES+ not downloaded (gated)")
    for lang in langs.TASK_LANGS[task]:
        rows = T.load_rows(task, lang)
        assert len(rows) == EXCEPTIONS.get((task, lang), SIZES[task]), (lang, len(rows))
        assert all(r["target"] for r in rows)
