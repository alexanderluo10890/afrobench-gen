"""Language lists of the card run (card authors' note, 2026-10-29, Appendix A).

Unit keys are the codes the datasets use: masakhane 3-letter codes for AfriMGSM, AfriMMLU,
AfriXNLI and Injongo; FLORES-200 codes for Belebele, SIB-200 and FLORES. Every task average
is the mean over ALL languages of its list (English and French included where listed).
"""

MASAKHANE = ["amh", "eng", "ewe", "fra", "hau", "ibo", "kin", "lin", "lug", "orm", "sna", "sot", "swa",
             "twi", "wol", "xho", "yor", "zul"]

TASK_LANGS = {
    "afrimgsm": MASAKHANE,                                             # 18
    "afrimmlu": MASAKHANE,                                             # 18
    "afrixnli": MASAKHANE,                                             # 18
    "injongo": [l for l in MASAKHANE if l != "fra"],                   # 17
    "belebele": ["afr_Latn", "amh_Ethi", "arz_Arab", "eng_Latn", "fra_Latn", "hau_Latn", "ibo_Latn",
                 "kin_Latn", "lin_Latn", "lug_Latn", "ary_Arab", "nya_Latn", "gaz_Latn", "plt_Latn",
                 "por_Latn", "sna_Latn", "som_Latn", "sot_Latn", "swh_Latn", "tir_Ethi", "tsn_Latn",
                 "wol_Latn", "xho_Latn", "yor_Latn", "zul_Latn"],      # 25
    "sib": ["afr_Latn", "amh_Ethi", "arz_Arab", "eng_Latn", "ewe_Latn", "hau_Latn", "ibo_Latn", "kin_Latn",
            "lin_Latn", "lug_Latn", "ary_Arab", "nya_Latn", "gaz_Latn", "plt_Latn", "por_Latn", "sna_Latn",
            "som_Latn", "sot_Latn", "swh_Latn", "tir_Ethi", "aeb_Arab", "twi_Latn", "wol_Latn", "xho_Latn",
            "yor_Latn", "zul_Latn"],                                   # 26
}

FLORES_LANGS = ["aeb_Arab", "afr_Latn", "amh_Ethi", "ary_Arab", "arz_Arab", "ewe_Latn", "gaz_Latn", "hau_Latn",
                "ibo_Latn", "kin_Latn", "lin_Latn", "lug_Latn", "nya_Latn", "sna_Latn", "som_Latn", "sot_Latn",
                "swh_Latn", "tir_Ethi", "tsn_Latn", "twi_Latn", "wol_Latn", "xho_Latn", "yor_Latn", "zul_Latn"]
# 24 languages x 2 directions = 48; FLORES score = mean of the 48 direction scores.
TASK_LANGS["flores"] = [f"{c}-eng_Latn" for c in FLORES_LANGS] + [f"eng_Latn-{c}" for c in FLORES_LANGS]

TASKS = ["afrimgsm", "afrimmlu", "afrixnli", "belebele", "injongo", "sib", "flores"]

# Language names used in the FLORES prompt (note, Appendix B).
FLORES_NAMES = {
    "eng_Latn": "English", "aeb_Arab": "Tunisian Arabic", "afr_Latn": "Afrikaans", "amh_Ethi": "Amharic",
    "ary_Arab": "Moroccan Arabic", "arz_Arab": "Egyptian Arabic", "ewe_Latn": "Ewe",
    "gaz_Latn": "West Central Oromo", "hau_Latn": "Hausa", "ibo_Latn": "Igbo", "kin_Latn": "Kinyarwanda",
    "lin_Latn": "Lingala", "lug_Latn": "Luganda", "nya_Latn": "Chichewa", "sna_Latn": "Shona",
    "som_Latn": "Somali", "sot_Latn": "Sesotho", "swh_Latn": "Swahili", "tir_Ethi": "Tigrinya",
    "tsn_Latn": "Setswana", "twi_Latn": "Twi", "wol_Latn": "Wolof", "xho_Latn": "Xhosa",
    "yor_Latn": "Yoruba", "zul_Latn": "Zulu",
}

# FLORES+ file for each code. FLORES+ splits Twi into Akuapem and Asante; FLORES-200's
# twi_Latn was Akuapem, so the note's `twi_Latn` maps to the Akuapem file.
FLORES_PLUS_FILE = {c: c for c in FLORES_NAMES}
FLORES_PLUS_FILE["twi_Latn"] = "twi_Latn_akua1239"

# masakhane code -> FLORES-200 code (AfriMGSM instructions in configs/ are keyed by FLORES code)
TO_FLORES = {"amh": "amh_Ethi", "eng": "eng_Latn", "ewe": "ewe_Latn", "fra": "fra_Latn", "hau": "hau_Latn",
             "ibo": "ibo_Latn", "kin": "kin_Latn", "lin": "lin_Latn", "lug": "lug_Latn", "orm": "gaz_Latn",
             "sna": "sna_Latn", "sot": "sot_Latn", "swa": "swh_Latn", "twi": "twi_Latn", "wol": "wol_Latn",
             "xho": "xho_Latn", "yor": "yor_Latn", "zul": "zul_Latn"}

DISPLAY = {"amh": "Amharic", "eng": "English", "ewe": "Ewe", "fra": "French", "hau": "Hausa", "ibo": "Igbo",
           "kin": "Kinyarwanda", "lin": "Lingala", "lug": "Luganda", "orm": "Oromo", "sna": "Shona",
           "sot": "Sesotho", "swa": "Swahili", "twi": "Twi", "wol": "Wolof", "xho": "Xhosa", "yor": "Yoruba",
           "zul": "Zulu", "fra_Latn": "French", "por_Latn": "Portuguese", "plt_Latn": "Malagasy",
           **FLORES_NAMES}


def display(code: str) -> str:
    if "-" in code:
        a, b = code.split("-")
        return f"{display(a)} → {display(b)}"
    return DISPLAY.get(code, code)
