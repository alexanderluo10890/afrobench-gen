"""Model registry and the tokenizer fix (the tokenizer check itself runs on the cluster)."""
import json
import re

from afbgen import config, models

CARD_KEYS = {"AfriMGSM", "AfriMMLU", "AfriXNLI", "Belebele", "Injongo", "SIB-200", "FLORES", "Overall"}


def test_registry_entries_are_pinned():
    reg = models.registry()
    assert models.DEFAULT in reg
    for name, m in reg.items():
        assert re.fullmatch(r"[0-9a-f]{40}", m["revision"]), name
        assert "/" in m["repo"], name
        assert m.get("tokenizer_fix") in (None, "tokenizer_json"), name
        if m.get("card"):
            assert set(m["card"]) == CARD_KEYS, name


def test_default_model_is_the_card_model():
    m = models.get(models.DEFAULT)
    assert m["repo"] == "McGill-NLP/AfriqueQwen3.5-4B-Instruct-v2"
    assert m["revision"].startswith("d9c3c6a")
    assert m["card"]["Overall"] == 61.3


def test_tiny_aya_gets_the_tokenizer_fix():
    assert models.get("tiny-aya-earth")["tokenizer_fix"] == "tokenizer_json"


def test_apply_tokenizer_fix(tmp_path, monkeypatch):
    monkeypatch.setenv("AFB_DATA_DIR", str(tmp_path))
    d = models.model_dir("tiny-aya-earth")
    d.mkdir(parents=True)
    (d / "tokenizer.json").write_text("{}")
    (d / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "CohereTokenizerFast", "x": 1}))
    models.apply_tokenizer_fix("tiny-aya-earth")
    fixed = models.tokenizer_path("tiny-aya-earth")
    assert fixed == d / "tokenizer-fixed"
    cfg = json.loads((fixed / "tokenizer_config.json").read_text())
    assert cfg == {"tokenizer_class": "PreTrainedTokenizerFast", "x": 1}
    assert (fixed / "tokenizer.json").exists()
    # models without a fix use their own directory
    assert models.tokenizer_path("qwen3.5-4b") == models.model_dir("qwen3.5-4b")
