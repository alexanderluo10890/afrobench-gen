import pytest

from afbgen import tasklib as T

OPTS4 = ["5", "7", "8", "9"]


@pytest.mark.parametrize("resp,strict,flex", [
    ("C", "C", "C"),
    ("C.", "C", "C"),
    (" (C) ", "C", "C"),
    ("**C**", "C", "C"),
    ("C. 8", "C", "C"),
    ("C\n\nExplanation: ...", "C", "C"),
    ("The answer is C.", "", "C"),
    ("Answer: (B)", "", "B"),
    ("8", "", "C"),            # option text instead of letter
    ("E", "", ""),             # not a valid option
    ("Ce", "", ""),            # a word, not a letter
    ("", "", ""),
])
def test_letters(resp, strict, flex):
    assert T.extract_letter_strict(resp, 4) == strict
    assert T.extract_letter_flexible(resp, OPTS4) == flex


@pytest.mark.parametrize("resp,num", [
    (r"so the answer is \boxed{18}", "18"),
    (r"\boxed{1,200} dollars", "1200"),
    (r"\boxed{\$70,000}", "70000"),
    (r"x = \boxed{\frac{3}{4}}... no wait \boxed{12}", "12"),
    (r"\boxed{12} and later the number 99", "12"),
    ("first 3 then 5 then 42.", "42"),
    ("answer 7.50", "7.5"),
    ("answer 18.0", "18"),
    ("-5 degrees", "-5"),
    ("no number here", ""),
])
def test_math(resp, num):
    assert T.extract_math(resp) == num


def test_math_scoring():
    assert T.process_math({"target": "70000"}, ["70000"])["acc"] == 1.0
    assert T.process_math({"target": "18"}, ["18"])["acc"] == 1.0
    assert T.process_math({"target": "18"}, [""])["acc"] == 0.0


@pytest.mark.parametrize("resp,strict,flex", [
    ("alarm", "alarm", "alarm"),
    ("Alarm.", "alarm", "alarm"),
    ("`bill_balance`", "bill_balance", "bill_balance"),
    ("The intent is bill_balance", "", "bill_balance"),   # not 'balance'
    ("timezone", "timezone", "timezone"),                 # not 'time'
    ("something else", "", ""),
])
def test_injongo(resp, strict, flex):
    assert T.extract_label_strict(resp, T.INJONGO_LABELS) == strict
    assert T.extract_label_flexible(resp, T.INJONGO_LABELS) == flex


def test_translation_cleanup():
    assert T.clean_translation("English: Hello there.\n\nNote: ...", "English") == "Hello there."
    assert T.clean_translation("  Hello there.  ", "English") == "Hello there."


def test_letter_list():
    assert T._letter_list(3) == "A, B, or C"
    assert T._letter_list(4) == "A, B, C, or D"
    assert T._letter_list(7) == "A, B, C, D, E, F, or G"


def test_card_mc_template_exact():
    doc = {"question": "Q?", "options": ["a", "b", "c", "d"]}
    assert T.text_mc(doc) == (
        "Answer the multiple-choice question by replying with ONLY the letter\n"
        "(A, B, C, or D) of the correct answer.\n\nQuestion: Q?\n\nA. a\nB. b\nC. c\nD. d\n\nAnswer:")


def test_corpus_metrics():
    pytest.importorskip("sacrebleu")
    items = [("the cat sat", "the cat sat"), ("a dog", "a dog")]
    assert T.agg_chrfpp(items) == pytest.approx(100.0)
