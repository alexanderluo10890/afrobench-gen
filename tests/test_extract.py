"""Prompts and extraction rules of the card run (card authors' note, 2026-10-29)."""
import re

import pytest

from afbgen import tasklib as T


def lm_eval_regex(task, reply):
    """What lm-eval's `regex` filter (group_select 0, fallback "[invalid]") returns."""
    m = re.findall(T.ANSWER_PATTERNS[task], reply)
    return (m[0] if isinstance(m[0], str) else next(x for x in m[0] if x)).strip() if m else "[invalid]"


# ---------------------------------------------------------------- prompts (Appendix B, verbatim)

def test_prompt_afrimmlu():
    d = {"question": "Q?", "A": "a", "B": "b", "C": "c", "D": "d"}
    assert T.text_afrimmlu(d) == (
        "Answer the multiple-choice question by replying with ONLY the letter\n"
        "(A, B, C, or D) of the correct answer.\n\nQuestion: Q?\n\nA. a\nB. b\nC. c\nD. d\n\nAnswer:")


def test_prompt_afrixnli():
    assert T.text_afrixnli({"premise": "P.", "hypothesis": "H."}) == (
        "Determine the relationship between the premise and the hypothesis. Reply with ONLY the\n"
        "letter of the correct option: A (entailment \u2014 the hypothesis follows from the premise),\n"
        "B (neutral \u2014 neither follows nor contradicts), or C (contradiction \u2014 the hypothesis\n"
        "contradicts the premise).\n\nPremise: P.\nHypothesis: H.\n\n"
        "A. Entailment\nB. Neutral\nC. Contradiction\n\nAnswer:")


def test_prompt_belebele():
    d = {"passage": "Text.", "question": "Q?", "A": "a", "B": "b", "C": "c", "D": "d"}
    assert T.text_belebele(d) == (
        "Read the passage and answer the multiple-choice question by replying with ONLY the\n"
        "letter (A, B, C, or D) of the correct answer.\n\nPassage: Text.\n\nQuestion: Q?\n\n"
        "A. a\nB. b\nC. c\nD. d\n\nAnswer:")


def test_prompt_injongo():
    p = T.text_injongo({"text": "wake me at 6"})
    assert p.startswith("Classify the intent of the following request. Reply with ONLY the number\n"
                        "(1-40) of the correct intent from the list.\n\nRequest: wake me at 6\n\nIntents:\n1. alarm\n2. balance\n")
    assert "\n25. restaurant_reservation\n" in p and p.endswith("\n39. update_playlist\n40. weather\n\nAnswer:")


def test_prompt_sib():
    assert T.text_sib({"text": "T."}) == (
        "Classify the topic of the following text. Reply with ONLY the letter\n"
        "(A, B, C, D, E, F, or G) of the correct topic.\n\nText: T.\n\n"
        "A. Entertainment\nB. Geography\nC. Health\nD. Politics\nE. Science/Technology\nF. Sports\nG. Travel\n\nAnswer:")


def test_prompt_translation():
    assert T.text_translation({"src_name": "Hausa", "tgt_name": "English", "source": "Sannu."}) == (
        "Translate the following sentence from Hausa to English.\n\nHausa: Sannu.\nEnglish:")


def test_prompt_math_is_user_message_text():
    assert T.text_math("Instr.", "Q?") == "Instr.\n\nQ?"


def test_sib_injongo_mappings():
    assert T.SIB_CATEGORIES == sorted(T.SIB_CATEGORIES)
    assert T.LETTERS[T.SIB_CATEGORIES.index("science/technology")] == "E"
    assert T.INJONGO_INTENTS.index("restaurant_reservation") + 1 == 25


# ---------------------------------------------------------------- MC / Injongo extraction (4.3)

@pytest.mark.parametrize("task,reply,want", [
    ("afrimmlu", "B", "B"),
    ("afrimmlu", "B.", "B"),
    ("afrimmlu", "The answer is C.", "C"),
    ("afrimmlu", "A seems right, but the answer is D", "D"),   # LAST standalone letter
    ("afrimmlu", "E", "[invalid]"),
    ("afrimmlu", "", "[invalid]"),
    ("afrixnli", "C. Contradiction", "C"),
    ("afrixnli", "D", "[invalid]"),
    ("sib", "G. Travel", "G"),
    ("injongo", "25", "25"),
    ("injongo", "Answer: 7", "7"),
    ("injongo", "40. weather", "40"),
    ("injongo", "41", "[invalid]"),
    ("injongo", "1. alarm or 32. time", "32"),
])
def test_last_option(task, reply, want):
    assert lm_eval_regex(task, reply) == want


# ---------------------------------------------------------------- AfriMGSM extraction (4.2)

@pytest.mark.parametrize("out,want", [
    (r"so \boxed{18}", 18.0),
    (r"\boxed{1,200} dollars", 1200.0),
    (r"\boxed{\$70,000}", 70000.0),
    (r"x \boxed{\frac{3}{4}} no wait \boxed{12}", 12.0),
    (r"\boxed{}", None),                                    # a box without a number is wrong
    (r"\boxed{\text{eighteen}} and 18 later", None),        # last box decides, even without a number
    ("first 3 then 5 then 42.", 42.0),                      # no box: last number
    ("answer \u221212", -12.0),                             # Unicode minus
    ("total 1\u00a0000", 1000.0),                           # no-break space between digits
    ("total 1\u2009000", 1000.0),                           # thin space between digits
    ("days 3\u20135, so 4", 4.0),                           # en dash is not a minus
    ("<think>maybe \\boxed{7}</think> The answer is 8.", 7.0),  # thinking is included; last box wins
    ("no number", None),
])
def test_math_extraction(out, want):
    assert T.extract_math(out) == want


def test_math_correct_tolerance():
    assert T.math_correct(r"\boxed{18.0000001}", "18")
    assert not T.math_correct(r"\boxed{18.01}", "18")
    assert not T.math_correct(r"\boxed{}", "18")


# ---------------------------------------------------------------- FLORES cleaning (4.4)

@pytest.mark.parametrize("reply,want", [
    ("  Hello there.  ", "Hello there."),
    ("Hello there.\n\nNote: ...", "Hello there."),
    ("English: Hello.", "English: Hello."),                 # no prefix removal
    ("<think>a</think>x</think> Hi.", "Hi."),               # after the LAST </think>
    ("line one\nline two", "line one\nline two"),           # single newlines are kept
    ("\n\nlate", ""),
])
def test_clean_translation(reply, want):
    assert T.clean_translation(reply) == want


def test_corpus_metrics():
    pytest.importorskip("sacrebleu")
    assert T.agg_chrfpp([("the cat sat", "the cat sat"), ("a dog", "a dog")]) == pytest.approx(100.0)
