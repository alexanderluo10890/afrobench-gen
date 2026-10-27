"""Generative AfroBench-Lite tasks with the AfriqueQwen3.5 model-card prompts.

Everything lm-eval needs (dataset loaders, prompt builders, answer extraction filters,
scoring, aggregation) lives here; the task YAMLs in tasks/afrobench_gen/ only reference
these functions through tasks/afrobench_gen/afb_utils.py.

Datasets are read from local snapshots made by `python -m afbgen.prepare` (pinned
revisions, see configs/pins.yaml), so evaluation never touches the network.
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

from afbgen import langs

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

XNLI_LABELS = ["entailment", "neutral", "contradiction"]  # dataset label ids 0, 1, 2
SIB_LABELS = ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]
INJONGO_LABELS = [
    "alarm", "balance", "bill_balance", "book_flight", "book_hotel", "calendar_update",
    "cancel_reservation", "car_rental", "confirm_reservation", "cook_time", "exchange_rate",
    "food_last", "freeze_account", "ingredients_list", "interest_rate", "international_visa",
    "make_call", "meal_suggestion", "min_payment", "pay_bill", "pin_change", "play_music",
    "plug_type", "recipe", "restaurant_reservation", "restaurant_reviews", "restaurant_suggestion",
    "share_location", "shopping_list_update", "spending_history", "text", "time", "timezone",
    "transactions", "transfer", "translate", "travel_notification", "travel_suggestion",
    "update_playlist", "weather",
]


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


def _mc_doc(question: str, options: list[str], answer_idx: int, **extra) -> dict:
    return {"question": question, "options": list(options), "answer_idx": int(answer_idx),
            "target": LETTERS[int(answer_idx)], **extra}


def load_rows(task: str, lang: str) -> list[dict]:
    """Return the test split for (task, lang) as a list of normalized docs."""
    d = data_dir(task)
    if task == "afrimgsm":
        return [{"question": r["question"], "target": r["answer_number"].strip(), "lang": lang}
                for r in _read_tsv(d / "data" / lang / "test.tsv")]
    if task == "afrimmlu":
        rows = []
        for r in _read_tsv(d / "data" / lang / "test.tsv"):
            options = ast.literal_eval(r["choices"])
            rows.append(_mc_doc(r["question"], options, LETTERS.index(r["answer"].strip()),
                                subject=r["subject"], lang=lang))
        return rows
    if task == "afrixnli":
        rows = []
        for r in _read_tsv(d / "data" / lang / "test.tsv"):
            q = (f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}\n"
                 "What is the relationship between the premise and the hypothesis?")
            rows.append(_mc_doc(q, XNLI_LABELS, int(r["label"]), lang=lang))
        return rows
    if task == "belebele":
        rows = []
        for r in _read_jsonl(d / "data" / f"{langs.flores_code(lang)}.jsonl"):
            q = f"{r['flores_passage']}\n\n{r['question'].strip()}"
            options = [r[f"mc_answer{i}"] for i in range(1, 5)]
            rows.append(_mc_doc(q, options, int(r["correct_answer_num"]) - 1, lang=lang))
        return rows
    if task == "sib":
        rows = []
        for r in _read_tsv(d / "data" / langs.flores_code(lang) / "test.tsv"):
            q = f"What is the topic of the following text?\n\n{r['text']}"
            rows.append(_mc_doc(q, SIB_LABELS, SIB_LABELS.index(r["category"].strip()), lang=lang))
        return rows
    if task == "injongo":
        return [{"text": r["text"], "target": r["intent"].strip(), "lang": lang}
                for r in _read_jsonl(d / lang / "test.jsonl")]
    if task == "flores":
        src, tgt = lang.split("-")
        src_s, tgt_s = _flores_side(d, src), _flores_side(d, tgt)
        assert len(src_s) == len(tgt_s), (lang, len(src_s), len(tgt_s))
        return [{"source": s, "target": t, "src_name": langs.name(src), "tgt_name": langs.name(tgt),
                 "lang": lang} for s, t in zip(src_s, tgt_s)]
    raise KeyError(task)


def _flores_side(d: Path, lang: str) -> list[str]:
    import pandas as pd

    files = sorted((d / "data" / "language" / langs.flores_code(lang)).glob("devtest-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no FLORES devtest parquet for {lang} under {d}")
    df = pd.concat([pd.read_parquet(f) for f in files])
    if "id" in df.columns:
        df = df.sort_values("id")
    col = "sentence" if "sentence" in df.columns else "text"
    return df[col].tolist()


def make_loader(task: str):
    def _load(lang: str, **_ignored):
        return _as_dataset(load_rows(task, lang))
    _load.__name__ = f"load_{task}"
    return _load


load_afrimgsm = make_loader("afrimgsm")
load_afrimmlu = make_loader("afrimmlu")
load_afrixnli = make_loader("afrixnli")
load_belebele = make_loader("belebele")
load_sib = make_loader("sib")
load_injongo = make_loader("injongo")
load_flores = make_loader("flores")


# --------------------------------------------------------------------------- prompts
# Card "Multiple choice" template, generalized from 4 to N options:
# the letter list "(A, B, C, or D)" and the option lines follow the number of options.

def _letter_list(n: int) -> str:
    ls = list(LETTERS[:n])
    return ls[0] if n == 1 else ", ".join(ls[:-1]) + (", or " if n > 2 else " or ") + ls[-1]


def text_mc(doc: dict) -> str:
    n = len(doc["options"])
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(doc["options"]))
    return (
        "Answer the multiple-choice question by replying with ONLY the letter\n"
        f"({_letter_list(n)}) of the correct answer.\n\n"
        f"Question: {doc['question']}\n\n"
        f"{opts}\n\n"
        "Answer:"
    )


def text_math(doc: dict) -> str:
    # The instruction is in the system prompt (configs/math_system_prompts.yaml);
    # the user turn is the bare question, as in the card's usage example.
    return doc["question"]


def text_translation(doc: dict) -> str:
    # Card "Translation" template.
    return (
        f"Translate the following sentence from {doc['src_name']} to {doc['tgt_name']}.\n\n"
        f"{doc['src_name']}: {doc['source']}\n"
        f"{doc['tgt_name']}:"
    )


def text_injongo(doc: dict) -> str:
    # Not in the card (40 labels do not fit the A-D template): label-output variant
    # in the same style. See README "Decisions to confirm".
    return (
        "Classify the intent of the text by replying with ONLY one label from the list below.\n\n"
        f"Text: {doc['text']}\n\n"
        f"Labels: {', '.join(INJONGO_LABELS)}\n\n"
        "Answer:"
    )


def target(doc: dict) -> str:
    return doc["target"]


# --------------------------------------------------------------------------- extraction

_STRIP = " \t\n\r*_`'\"“”‘’"


def _strip_md(s: str) -> str:
    return s.strip().strip(_STRIP).strip()


def extract_letter_strict(resp: str, n: int) -> str:
    """The response must start with the letter, e.g. 'B', 'B.', '(B)', '**B**', 'B) text'."""
    s = _strip_md(resp)
    m = re.match(r"^\(?([A-Z])(?![A-Za-z0-9])", s)
    if m and m.group(1) in LETTERS[:n]:
        return m.group(1)
    return ""


def extract_letter_flexible(resp: str, options: list[str]) -> str:
    n = len(options)
    if (x := extract_letter_strict(resp, n)):
        return x
    valid = LETTERS[:n]
    for pat in (r"answer\s*(?:is|:)?\s*:?\s*\**\(?([A-Z])\)?(?![A-Za-z0-9])",
                r"^\s*(?:option|choice)\s*\(?([A-Z])\)?(?![A-Za-z0-9])"):
        for m in re.finditer(pat, resp, flags=re.IGNORECASE | re.MULTILINE):
            if m.group(1).upper() in valid:
                return m.group(1).upper()
    s = _strip_md(resp).rstrip(".").strip().lower()
    for i, o in enumerate(options):
        if s == o.strip().lower():
            return LETTERS[i]
    return ""


_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def _norm_number(s: str) -> str:
    s = s.replace(",", "").replace("$", "").strip()
    try:
        v = float(s)
    except ValueError:
        return ""
    return str(int(v)) if v == int(v) else repr(v)


def _last_boxed(s: str) -> str | None:
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    k = j
    while k < len(s) and depth:
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
        k += 1
    return s[j:k - 1] if depth == 0 else s[j:]


def extract_math(resp: str) -> str:
    """Card rule: the number inside \\boxed{} or else the last number in the output."""
    boxed = _last_boxed(resp)
    if boxed is not None:
        nums = re.findall(_NUM, boxed.replace("\\,", "").replace("{,}", ","))
        if nums:
            return _norm_number(nums[-1])
    nums = re.findall(_NUM, resp)
    return _norm_number(nums[-1]) if nums else ""


def _norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", _strip_md(s).lower().rstrip(".").strip())


def extract_label_strict(resp: str, labels: list[str]) -> str:
    s = _norm_label(resp)
    return s if s in labels else ""


def extract_label_flexible(resp: str, labels: list[str]) -> str:
    if (x := extract_label_strict(resp, labels)):
        return x
    low = resp.lower()
    best = None  # earliest occurrence; longest label wins ties (bill_balance over balance)
    for lab in labels:
        m = re.search(rf"(?<![a-z_]){re.escape(lab)}(?![a-z_])", low)
        if m and (best is None or (m.start(), -len(lab)) < (best[0], -len(best[1]))):
            best = (m.start(), lab)
    return best[1] if best else ""


def clean_translation(resp: str, tgt_name: str) -> str:
    s = resp.strip()
    s = re.sub(rf"^{re.escape(tgt_name)}\s*:\s*", "", s, flags=re.IGNORECASE)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines[0] if lines else ""


# lm-eval `custom` filters: (resps: list[list[str]], docs) -> list[list[str]]

def _fmap(fn):
    def _f(resps, docs):
        return [[fn(r, d) for r in rs] for rs, d in zip(resps, docs)]
    return _f


filter_mc_strict = _fmap(lambda r, d: extract_letter_strict(r, len(d["options"])))
filter_mc_flexible = _fmap(lambda r, d: extract_letter_flexible(r, d["options"]))
filter_math = _fmap(lambda r, d: extract_math(r))
filter_injongo_strict = _fmap(lambda r, d: extract_label_strict(r, INJONGO_LABELS))
filter_injongo_flexible = _fmap(lambda r, d: extract_label_flexible(r, INJONGO_LABELS))
filter_translation = _fmap(lambda r, d: clean_translation(r, d["tgt_name"]))


# --------------------------------------------------------------------------- scoring

def process_exact(doc: dict, results: list[str]) -> dict:
    return {"acc": float(results[0] == doc["target"])}


def process_math(doc: dict, results: list[str]) -> dict:
    return {"acc": float(results[0] != "" and results[0] == _norm_number(doc["target"]))}


def process_translation(doc: dict, results: list[str]) -> dict:
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
