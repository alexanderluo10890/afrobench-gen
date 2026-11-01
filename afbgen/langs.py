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


# ---------------------------------------------------------------------------
# Extra benchmarks (not on the card; settings are ours, see README "Extra benchmarks").
# Language lists = every language of the dataset at the pinned revision.
EXTRA_TASKS = ["globalpiqa", "mmluprox"]
GLOBALPIQA_LANGS = [
    "acm_arab", "acq_arab", "aeb_arab", "afb_arab", "als_latn", "amh_ethi", "apc_arab_jord",
    "apc_arab_leba", "apc_arab_pale", "apc_arab_syri", "arb_arab", "arq_arab", "ars_arab",
    "ary_arab", "arz_arab", "asm_beng", "azj_latn", "bam_latn", "bcc_arab", "bel_cyrl", "ben_beng",
    "ben_latn", "bgc_deva", "bho_deva", "bos_latn", "bra_deva", "bsk_arab", "btx_latn", "bul_cyrl",
    "cat_latn", "ces_latn", "ckb_arab", "ckm_latn", "cls_deva", "cmn_hans", "cmn_hant", "dan_latn",
    "deu_latn", "dhd_deva", "dje_latn", "ekk_latn", "ekp_latn", "ell_grek", "eng_latn", "eus_latn",
    "fao_latn", "fin_latn", "fra_latn_cana", "fra_latn_fran", "glg_latn", "guj_gujr", "hau_latn",
    "haw_latn", "heb_hebr", "hin_deva", "hin_latn", "hrv_latn", "hun_latn", "hye_armn", "ibo_latn",
    "idu_latn", "ind_latn", "isl_latn", "iso_latn", "ita_latn", "jav_latn", "jpn_jpan", "kan_knda",
    "kan_latn", "kat_geor", "kaz_cyrl", "kin_latn", "kir_cyrl", "kor_hang", "lin_latn", "lit_latn",
    "luo_latn", "mal_mlym", "mar_deva", "mkd_cyrl", "mni_beng", "mni_mtei", "nag_latn", "nld_latn",
    "nno_latn", "nob_latn", "npi_deva", "ory_orya", "pan_guru", "pcm_latn", "pes_arab", "plt_latn",
    "pol_latn", "por_latn_braz", "por_latn_port", "ron_latn", "rus_cyrl", "rwr_deva", "sin_latn",
    "sin_sinh", "slk_latn", "slk_latn_sari", "slv_latn", "slv_latn_cerk", "slv_latn_prle",
    "snd_arab", "snd_deva", "spa_latn_mexi", "spa_latn_peru", "spa_latn_spai", "srp_cyrl",
    "srp_cyrl_torl", "srp_latn", "srp_latn_torl", "sun_latn", "swe_latn", "swh_latn", "swv_deva",
    "tam_latn", "tam_taml", "tel_latn", "tel_telu", "tgl_latn", "tha_thai", "tur_latn", "uig_arab",
    "ukr_cyrl", "urd_arab", "urd_latn", "urh_latn", "uzn_latn", "vie_latn", "yor_latn", "yue_hant",
    "zsm_latn", "zul_latn",
]
MMLUPROX_LANGS = [
    "af", "ar", "bn", "cs", "de", "en", "es", "fr", "hi", "hu", "id", "it", "ja", "ko", "mr", "ne",
    "pt", "ru", "sr", "sw", "te", "th", "uk", "ur", "vi", "wo", "yo", "zh", "zu",
]
TASK_LANGS["globalpiqa"] = GLOBALPIQA_LANGS     # 136
TASK_LANGS["mmluprox"] = MMLUPROX_LANGS         # 29

# African-language subsets reported next to the all-language averages (our grouping;
# includes the North-African Arabic varieties aeb/arq/ary/arz).
AFRICAN_SUBSET = {
    "globalpiqa": ['aeb_arab', 'amh_ethi', 'arq_arab', 'ary_arab', 'arz_arab', 'bam_latn', 'dje_latn', 'ekp_latn', 'hau_latn', 'ibo_latn', 'idu_latn', 'iso_latn', 'kin_latn', 'lin_latn', 'luo_latn', 'pcm_latn', 'plt_latn', 'swh_latn', 'urh_latn', 'yor_latn', 'zul_latn'],
    "mmluprox": ['af', 'sw', 'wo', 'yo', 'zu'],
}


def display(code: str) -> str:
    if "-" in code:
        a, b = code.split("-")
        return f"{display(a)} → {display(b)}"
    return DISPLAY.get(code, code)
