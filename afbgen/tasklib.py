"""AfroBench-Lite as generative tasks, with the settings of the card run.

Source of every prompt, mapping and rule below: the card authors' note "How to match the
AfriqueQwen3.5-4B-Instruct-v2 card numbers" (2026-10-29), sections 4.2-4.4 and Appendix B.

Datasets are read from local snapshots made by `python -m afbgen.prepare` (pinned
revisions, configs/pins.yaml), so evaluation never touches the network.
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

from afbgen import langs

LETTERS = "ABCDEFG"

# Injongo: the 40 intents, alphabetical, numbered 1-40; the target is the number.
INJONGO_INTENTS = [
    "alarm", "balance", "bill_balance", "book_flight", "book_hotel", "calendar_update",
    "cancel_reservation", "car_rental", "confirm_reservation", "cook_time", "exchange_rate",
    "food_last", "freeze_account", "ingredients_list", "interest_rate", "international_visa",
    "make_call", "meal_suggestion", "min_payment", "pay_bill", "pin_change", "play_music",
    "plug_type", "recipe", "restaurant_reservation", "restaurant_reviews", "restaurant_suggestion",
    "share_location", "shopping_list_update", "spending_history", "text", "time", "timezone",
    "transactions", "transfer", "translate", "travel_notification", "travel_suggestion",
    "update_playlist", "weather",
]
assert INJONGO_INTENTS == sorted(INJONGO_INTENTS) and len(INJONGO_INTENTS) == 40

# SIB-200: categories in alphabetical order -> A..G (dataset value -> prompt label)
SIB_CATEGORIES = ["entertainment", "geography", "health", "politics", "science/technology", "sports", "travel"]
SIB_LABELS = ["Entertainment", "Geography", "Health", "Politics", "Science/Technology", "Sports", "Travel"]

# lm-eval `regex` filter patterns (group_select 0, then take_first): the LAST standalone option.
ANSWER_PATTERNS = {
    "afrimmlu": r"(?s).*\b([A-D])\b",
    "afrixnli": r"(?s).*\b([A-C])\b",
    "belebele": r"(?s).*\b([A-D])\b",
    "sib": r"(?s).*\b([A-G])\b",
    "injongo": r"(?s).*\b([1-9]|[1-3][0-9]|40)\b",
    # extra benchmarks (not on the card): same rule
    "globalpiqa": r"(?s).*\b([A-B])\b",
    "mmluprox": r"(?s).*\b([A-J])\b",
}


# --------------------------------------------------------------------------- data

def data_dir(task: str) -> Path:
    root = os.environ.get("AFB_DATA_DIR")
    if not root:
        raise RuntimeError("AFB_DATA_DIR is not set (run `python -m afbgen.prepare` first)")
    return Path(root) / "datasets" / task


def _read_tsv(path: Path) -> list[dict]:
    # Same parser as the HF `datasets` csv builder (pandas, default quoting): some fields
    # are quoted and contain newlines.
    import pandas as pd

    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).to_dict("records")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _as_dataset(rows: list[dict]):
    import datasets

    return datasets.DatasetDict({"test": datasets.Dataset.from_list(rows)})


def load_rows(task: str, lang: str) -> list[dict]:
    """Test split for (task, lang) as normalized docs; every doc has a string `target`."""
    d = data_dir(task)
    if task == "afrimgsm":
        return [{"question": r["question"], "target": r["answer_number"].strip(), "lang": lang}
                for r in _read_tsv(d / "data" / lang / "test.tsv")]
    if task == "afrimmlu":
        rows = []
        for r in _read_tsv(d / "data" / lang / "test.tsv"):
            opts = ast.literal_eval(r["choices"])
            rows.append({"question": r["question"], "A": opts[0], "B": opts[1], "C": opts[2], "D": opts[3],
                         "target": r["answer"].strip(), "lang": lang})
        return rows
    if task == "afrixnli":
        return [{"premise": r["premise"], "hypothesis": r["hypothesis"], "target": "ABC"[int(r["label"])],
                 "lang": lang} for r in _read_tsv(d / "data" / lang / "test.tsv")]
    if task == "belebele":
        return [{"passage": r["flores_passage"], "question": r["question"], "A": r["mc_answer1"],
                 "B": r["mc_answer2"], "C": r["mc_answer3"], "D": r["mc_answer4"],
                 "target": "ABCD"[int(r["correct_answer_num"]) - 1], "lang": lang}
                for r in _read_jsonl(d / "data" / f"{lang}.jsonl")]
    if task == "sib":
        return [{"text": r["text"], "target": LETTERS[SIB_CATEGORIES.index(r["category"].strip())], "lang": lang}
                for r in _read_tsv(d / "data" / lang / "test.tsv")]
    if task == "injongo":
        # the card loader skips rows whose intent is not one of the 40 intents in the prompt
        return [{"text": r["text"], "target": str(INJONGO_INTENTS.index(r["intent"].strip()) + 1),
                 "intent": r["intent"].strip(), "lang": lang}
                for r in _read_jsonl(d / lang / "test.jsonl") if r["intent"].strip() in INJONGO_INTENTS]
    if task == "globalpiqa":
        rows = _read_tsv(d / "data" / f"nonparallel_{lang}.tsv")
        return [{"question": r["prompt"], "options": [r["solution0"], r["solution1"]],
                 "target": "AB"[int(r["label"])], "lang": lang} for r in rows]
    if task == "mmluprox":
        import pandas as pd

        df = pd.concat(pd.read_parquet(f) for f in sorted((d / lang).glob("test-*.parquet")))
        rows = []
        for r in df.to_dict("records"):
            opts = [r[f"option_{i}"] for i in range(10)]
            opts = [o for o in opts if isinstance(o, str) and o not in ("", "nan")]  # missing ones are trailing
            rows.append({"question": r["question"], "options": opts, "target": r["answer"].strip(),
                         "category": r["category"], "lang": lang})
        return rows
    if task == "flores":
        src, tgt = lang.split("-")
        s, t = _flores_plus(d, src), _flores_plus(d, tgt)
        ids = sorted(set(s) & set(t))
        assert len(ids) == len(s) == len(t), (lang, len(s), len(t), len(ids))
        return [{"id": i, "source": s[i], "target": t[i], "src_name": langs.FLORES_NAMES[src],
                 "tgt_name": langs.FLORES_NAMES[tgt], "lang": lang} for i in ids]
    raise KeyError(task)


def _flores_plus(d: Path, code: str) -> dict:
    """FLORES+ devtest for one language: {sentence id: text}; directions are joined by id."""
    path = d / "devtest" / f"{langs.FLORES_PLUS_FILE[code]}.jsonl"
    return {int(r["id"]): r["text"] for r in _read_jsonl(path)}


def make_loader(task: str):
    def _load(lang: str, **_ignored):
        return _as_dataset(load_rows(task, lang))
    _load.__name__ = f"load_{task}"
    return _load


load_afrimmlu = make_loader("afrimmlu")
load_afrixnli = make_loader("afrixnli")
load_belebele = make_loader("belebele")
load_sib = make_loader("sib")
load_injongo = make_loader("injongo")
load_flores = make_loader("flores")
load_globalpiqa = make_loader("globalpiqa")
load_mmluprox = make_loader("mmluprox")


# --------------------------------------------------------------------------- prompts (Appendix B)

def text_afrimmlu(doc: dict) -> str:
    return (
        "Answer the multiple-choice question by replying with ONLY the letter\n"
        "(A, B, C, or D) of the correct answer.\n\n"
        f"Question: {doc['question']}\n\n"
        f"A. {doc['A']}\nB. {doc['B']}\nC. {doc['C']}\nD. {doc['D']}\n\n"
        "Answer:"
    )


def text_afrixnli(doc: dict) -> str:
    return (
        "Determine the relationship between the premise and the hypothesis. Reply with ONLY the\n"
        "letter of the correct option: A (entailment — the hypothesis follows from the premise),\n"
        "B (neutral — neither follows nor contradicts), or C (contradiction — the hypothesis\n"
        "contradicts the premise).\n\n"
        f"Premise: {doc['premise']}\n"
        f"Hypothesis: {doc['hypothesis']}\n\n"
        "A. Entailment\nB. Neutral\nC. Contradiction\n\n"
        "Answer:"
    )


def text_belebele(doc: dict) -> str:
    return (
        "Read the passage and answer the multiple-choice question by replying with ONLY the\n"
        "letter (A, B, C, or D) of the correct answer.\n\n"
        f"Passage: {doc['passage']}\n\n"
        f"Question: {doc['question']}\n\n"
        f"A. {doc['A']}\nB. {doc['B']}\nC. {doc['C']}\nD. {doc['D']}\n\n"
        "Answer:"
    )


def text_injongo(doc: dict) -> str:
    intents = "\n".join(f"{i}. {name}" for i, name in enumerate(INJONGO_INTENTS, 1))
    return (
        "Classify the intent of the following request. Reply with ONLY the number\n"
        "(1-40) of the correct intent from the list.\n\n"
        f"Request: {doc['text']}\n\n"
        f"Intents:\n{intents}\n\n"
        "Answer:"
    )


def text_sib(doc: dict) -> str:
    opts = "\n".join(f"{L}. {name}" for L, name in zip(LETTERS, SIB_LABELS))
    return (
        "Classify the topic of the following text. Reply with ONLY the letter\n"
        "(A, B, C, D, E, F, or G) of the correct topic.\n\n"
        f"Text: {doc['text']}\n\n"
        f"{opts}\n\n"
        "Answer:"
    )


def text_translation(doc: dict) -> str:
    return (
        f"Translate the following sentence from {doc['src_name']} to {doc['tgt_name']}.\n\n"
        f"{doc['src_name']}: {doc['source']}\n"
        f"{doc['tgt_name']}:"
    )


def _letter_list(n: int) -> str:
    ls = list("ABCDEFGHIJ"[:n])
    return " or ".join(ls) if n == 2 else ", ".join(ls[:-1]) + ", or " + ls[-1]


def text_mc_generic(doc: dict) -> str:
    """The card MC template with the letter list adapted to the number of options.
    Used for the extra benchmarks (Global-PIQA: 2 options, MMLU-ProX: up to 10)."""
    n = len(doc["options"])
    opts = "\n".join(f"{'ABCDEFGHIJ'[i]}. {o}" for i, o in enumerate(doc["options"]))
    return (
        "Answer the multiple-choice question by replying with ONLY the letter\n"
        f"({_letter_list(n)}) of the correct answer.\n\n"
        f"Question: {doc['question']}\n\n"
        f"{opts}\n\n"
        "Answer:"
    )


text_globalpiqa = text_mc_generic
text_mmluprox = text_mc_generic


def text_math(instruction: str, question: str) -> str:
    """AfriMGSM user message (4.2): instruction, a blank line, the question. No system message."""
    return f"{instruction}\n\n{question}"


def target(doc: dict) -> str:
    return doc["target"]


# --------------------------------------------------------------------------- AfriMGSM extraction (4.2)

_MINUS = dict.fromkeys(map(ord, "−﹣－"), "-")   # Unicode minus signs -> '-' (not dashes)
_SPACE_IN_NUMBER = re.compile(r"(?<=\d)[    ](?=\d)")  # no-break / thin spaces
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _last_boxed(s: str) -> str | None:
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j = k = i + len("\\boxed{")
    depth = 1
    while k < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[k], 0)
        k += 1
    return s[j:k - 1] if depth == 0 else s[j:]


def _to_float(num: str) -> float | None:
    try:
        return float(num.replace(",", ""))
    except ValueError:
        return None


def extract_math(output: str) -> float | None:
    """Card rule on the FULL output (thinking included):
    1. Unicode minus -> '-', drop no-break/thin spaces between digits;
    2. last \\boxed{...}: its last number; a box without a number is wrong (None);
    3. no box: the last number in the output; 4. thousands commas removed."""
    s = _SPACE_IN_NUMBER.sub("", output.translate(_MINUS))
    boxed = _last_boxed(s)
    nums = _NUMBER.findall(boxed if boxed is not None else s)
    return _to_float(nums[-1]) if nums else None


def math_correct(output: str, gold: str) -> bool:
    pred, ref = extract_math(output), _to_float(gold)
    return pred is not None and ref is not None and abs(pred - ref) <= 1e-6


# --------------------------------------------------------------------------- FLORES cleaning (4.4)

def clean_translation(reply: str) -> str:
    """After the last </think>; before the first blank line; stripped. No prefix removal."""
    if "</think>" in reply:
        reply = reply.rsplit("</think>", 1)[1]
    return reply.split("\n\n", 1)[0].strip()


def filter_translation(resps, docs):
    return [[clean_translation(r) for r in rs] for rs in resps]


def process_translation(doc: dict, results: list[str]) -> dict:
    # chrF++ / BLEU are secondary; the reported FLORES metric is SSA-COMET (afbgen.comet_score)
    pair = (doc["target"], results[0])
    return {"chrfpp": pair, "bleu": pair}


def agg_chrfpp(items) -> float:
    import sacrebleu

    refs, preds = zip(*items)
    return sacrebleu.corpus_chrf(list(preds), [list(refs)], word_order=2).score


def agg_bleu(items) -> float:
    import sacrebleu

    refs, preds = zip(*items)
    return sacrebleu.corpus_bleu(list(preds), [list(refs)]).score
