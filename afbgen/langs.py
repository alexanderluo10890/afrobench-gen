"""Language registry for the AfroBench-Lite language set.

Canonical key = the 3-letter code used by the masakhane datasets (AfriMGSM, AfriMMLU,
AfriXNLI, Injongo). FLORES-style codes are used by Belebele, SIB-200 and FLORES.
"""

LANGS = {
    # key: (English name, FLORES-200 code)
    "eng": ("English", "eng_Latn"),
    "amh": ("Amharic", "amh_Ethi"),
    "hau": ("Hausa", "hau_Latn"),
    "ibo": ("Igbo", "ibo_Latn"),
    "kin": ("Kinyarwanda", "kin_Latn"),
    "lin": ("Lingala", "lin_Latn"),
    "lug": ("Luganda", "lug_Latn"),
    "orm": ("Oromo", "gaz_Latn"),
    "sna": ("Shona", "sna_Latn"),
    "sot": ("Sesotho", "sot_Latn"),
    "swa": ("Swahili", "swh_Latn"),
    "xho": ("Xhosa", "xho_Latn"),
    "yor": ("Yoruba", "yor_Latn"),
    "zul": ("Zulu", "zul_Latn"),
    "fra": ("French", "fra_Latn"),  # FLORES only (AfroBench-Lite includes fra<->eng)
}

AFRICAN = ["amh", "hau", "ibo", "kin", "lin", "lug", "orm", "sna", "sot", "swa", "xho", "yor", "zul"]

# AfroBench-Lite language lists per benchmark (prompt_with_API/afrobench_lite/*.yaml).
# Note: afrobench_lite/sib.yaml lists "ssh_Latn", an obvious typo for swh_Latn (Swahili).
TASK_LANGS = {
    "afrimgsm": ["eng"] + AFRICAN,
    "afrimmlu": ["eng"] + AFRICAN,
    "afrixnli": ["eng"] + AFRICAN,
    "belebele": ["eng"] + AFRICAN,
    "injongo": ["eng"] + AFRICAN,
    "sib": ["eng"] + AFRICAN,
}

# FLORES directions are encoded as "<src>-<tgt>" with the 3-letter keys above.
FLORES_DIRECTIONS = [f"{l}-eng" for l in AFRICAN + ["fra"]] + [f"eng-{l}" for l in AFRICAN + ["fra"]]
TASK_LANGS["flores"] = FLORES_DIRECTIONS

TASKS = ["afrimgsm", "afrimmlu", "afrixnli", "belebele", "injongo", "sib", "flores"]


def name(lang: str) -> str:
    return LANGS[lang][0]


def flores_code(lang: str) -> str:
    return LANGS[lang][1]
