#!/usr/bin/env python3

import os
import json
import pickle
import joblib
import hashlib
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import uproot
import tensorflow as tf


# ============================================================
# User configuration
# ============================================================

current_date = datetime.now().strftime("%m%d")

# Point this to the completed all-era training output.
MODEL_DIR = "/eos/user/q/qguo/SWAN_projects/ML_test/noBr4_simple_dnn_DAandNoDomain_20260721_signedPhysW_both_v2/"
MODEL_DIR = "/eos/user/q/qguo/SWAN_projects/ML_test/noBr4_simple_dnn_DAandNoDomain_20260721_allYears_gpu_v1/"

# Which saved simple-DNN variant to apply:
#   "DA"       -> simple_DA_model_fold_N.keras, with task score and domain score
#   "noDomain" -> simple_noDomain_model_fold_N.keras, task score only
APPLY_TAG = "DA"
#APPLY_TAG = "noDomain"

N_FOLDS = 4

TREE_NAME_DEFAULT = "data_two_jet_m110To150"

TARGET_CATE_INDEX = 3
CATEGORY_BRANCH = "cate_index"

MASS_BRANCH = "diMufsr_kit_BSC_mass"
WEIGHT_BRANCH = "eventWeight"

# The model keeps one feature schema, while the physical ntuple names differ:
#   2022-2023: diMufsr_rc_BSC_*
#   2024-2025: diMufsr_kit_BSC_*
# Define both directions so this also supports older features.json files that
# stored rc-style names.
BRANCH_ALIASES = {
    "diMufsr_kit_BSC_mass": ["diMufsr_kit_BSC_mass", "diMufsr_rc_BSC_mass"],
    "diMufsr_kit_BSC_pt": ["diMufsr_kit_BSC_pt", "diMufsr_rc_BSC_pt"],
    "diMufsr_kit_BSC_eta": ["diMufsr_kit_BSC_eta", "diMufsr_rc_BSC_eta"],
    "log_diMufsr_kit_BSC_pt": ["log_diMufsr_kit_BSC_pt", "log_diMufsr_rc_BSC_pt"],
    "diMufsr_rc_BSC_mass": ["diMufsr_rc_BSC_mass", "diMufsr_kit_BSC_mass"],
    "diMufsr_rc_BSC_pt": ["diMufsr_rc_BSC_pt", "diMufsr_kit_BSC_pt"],
    "diMufsr_rc_BSC_eta": ["diMufsr_rc_BSC_eta", "diMufsr_kit_BSC_eta"],
    "log_diMufsr_rc_BSC_pt": ["log_diMufsr_rc_BSC_pt", "log_diMufsr_kit_BSC_pt"],
}

# These metadata branches are not uniform across samples. In 2022-2023 they
# are DY-only; in 2024-2025 they are absent from data.
OPTIONAL_BRANCH_DEFAULTS = {
    "genvbffilter_flag": -1,
    "n_jets_matched_genjet": -1,
}

SR_LOW = 115.0
SR_HIGH = 135.0

FIX_SB_MASS_TO_125 = True
FIXED_MASS_VALUE = 125.0

BATCH_SIZE = 16384

STRICT_TRAINING_FOLDS = True
ALLOW_HASH_FOR_TRAINING_SYSTS = True

# A reskim can differ from the training cache by a handful of events. Permit a
# deterministic hash fold only for a tiny unmatched tail; large mismatches
# still stop to protect out-of-fold evaluation from accidental leakage.
MAX_UNMATCHED_TRAINING_EVENTS_FOR_HASH = 10
MAX_UNMATCHED_TRAINING_FRACTION_FOR_HASH = 1.0e-5
ALWAYS_ALLOW_UNMATCHED_TRAINING_EVENTS_FOR_HASH = 2

# None means each application era uses its real source_year and era_* encoding.
# Set this to e.g. "2024" only when deliberately applying a 2024-only model
# to another year while keeping the model's 2024 year encoding.
SOURCE_YEAR_FOR_FEATURES = None

# Every era used by this shared model must use its saved exact out-of-fold map.
TRAINING_ERAS_FOR_EXACT_FOLDS = {
    "2022", "2022EE", "2023", "2023BPix", "2024", "2025",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply the trained simple DNN to nominal and systematic ROOT trees. "
            "Systematic tree names are expected as <nominal_tree>__<sysName>."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tree-mode",
        choices=["all", "nominal"],
        default="all",
        help="all: nominal plus <nominal>__* trees; nominal: only nominal tree.",
    )
    return parser.parse_args()


ARGS = parse_args()

# ============================================================
# Fold matching control
# ============================================================
# If task/domain pkl has run/lumi/event:
#     exact out-of-fold matching is used.
#
# If task/domain pkl does NOT have run/lumi/event:
#     if True  -> use deterministic random/hash fold as fallback
#     if False -> stop
# ============================================================

USE_RANDOM_HASH_FOLD_FOR_APPLICATION = True
#USE_RANDOM_HASH_FOLD_FOR_APPLICATION = False

RANDOM_FOLD_SEED_TAG = "simple_dnn_application_20260626_v1"
RANDOM_FOLD_SEED_TAG = "simple_dnn_application_20260626_v1_noDomain"
RANDOM_FOLD_SEED_TAG = "simple_dnn_application_20260701_v1_noDomain"
RANDOM_FOLD_SEED_TAG = "simple_dnn_application_20260626_v1_DA_2025"
RANDOM_FOLD_SEED_TAG = "simple_dnn_application_20260721_v1_DA"

# simple_dnn_da_comparison.py trains model fold N on all events except
# fold_id == N. Therefore fold_id == N must be applied with model fold N.
APPLICATION_MODEL_FOR_EVENT_FOLD = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
}


# ============================================================
# Samples used in training
# ============================================================

# SR task training samples:
# SR signal + SR backgrounds used in task loss.
TASK_TRAINING_SAMPLES = {
    #"dy_inc",
    "dy_inc_fail",
    "dy_vbffilter",
    #"ewk_zjj",
    "TTTo2L2Nu",
    "vbf_hmm",
    #"ggh_hmm",
}

# SB domain training samples:
# SB data + SB background MC used in domain loss.
DOMAIN_TRAINING_SAMPLES = {
    "data",
    #"dy_inc",
    "dy_inc_fail",
    "dy_vbffilter",
    #"ewk_zjj",
    "TTTo2L2Nu",
}


# ============================================================
# Input paths and sample list
# ============================================================

path = {}
path[0] = "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2022_ggHVBF/"
path[1] = "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2022EE_ggHVBF/"
path[2] = "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2023_ggHVBF/"
path[3] = "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2023BPix_ggHVBF/"
#path[4] = "/eos/user/z/zhangxu/sharing/hmm/2024_v4/skimmed_ntuples/SRSB_v2/"
#path[5] = "/eos/user/h/hakou/Hmumu_Share/qguo/2025/skimmed_ntuples/SRSB_noJetHornVeto/"
#path[4] = "/eos/user/h/hakou/Hmumu_Share/qguo/2024_v5_SRSB/skimmed_ntuples/SRSB_Nor/"
#path[5] = "/eos/user/h/hakou/Hmumu_Share/qguo/2025_v5_SRSB_v2/skimmed_ntuples/SRSB_Nor/"
path[4] = "/eos/user/h/hakou/Hmumu_Share/qguo/2024_v5_SRSB/skimmed_ntuples/SRSB/"
path[5] = "/eos/user/h/hakou/Hmumu_Share/qguo/2025_v5_SRSB_v2/skimmed_ntuples/SRSB/"

era_names = {
    0: "2022",
    1: "2022EE",
    2: "2023",
    3: "2023BPix",
    4: "2024",
    5: "2025",
}

tree_name_by_path = {
    0: "data_two_jet_m110To150_VBF",
    1: "data_two_jet_m110To150_VBF",
    2: "data_two_jet_m110To150_VBF",
    3: "data_two_jet_m110To150_VBF",
    4: "data_two_jet_m110To150_VBF",
    5: "data_two_jet_m110To150_VBF",
}

samples_by_path = {
    0: {
        #"data": "data.root",
        #"dy_inc": "DY_105To160.root",
        #"dy_inc_fail": "DY_105To160_Inc_failvbffilter.root",
        #"dy_vbffilter": "DY_105To160_Fil-VBF_passvbffilter.root",
        #"ewk_zjj": "EWK_LLJJ_M105To160.root",
        #"ewk_zjj_pythia": "EWK_2Mu2J_105to160_pythia.root",
        #"TTTo2L2Nu": "TTTo2L2Nu.root",
        #"vbf_hmm": "VBFHToMuMu_M125_ggHUnc.root",
        #"ggh_hmm": "GluGluHToMuMu_M125_ggHUnc.root",
        #"WWTo2L2Nu": "WWTo2L2Nu.root",
        #"WWW": "WWW.root",
        #"WWZ": "WWZ.root",
        #"WZTo2L2Q": "WZTo2L2Q.root",
        #"WZTo3LNu": "WZTo3LNu.root",
        #"WZZ": "WZZ.root",
        #"ZZTo2L2Nu": "ZZTo2L2Nu.root",
        #"ZZTo2L2Q": "ZZTo2L2Q.root",
        #"ZZTo4L": "ZZTo4L.root",
        #"ZZZ": "ZZZ.root",
        #"ST_tW_antitop": "ST_tW_antitop.root",
        #"ST_tW_top": "ST_tW_top.root",
        "vh_hmm": "VHToMuMu_M125.root",
        "tth_hmm": "TTHto2Mu_M-125.root",
        "rare_hmm": "rareHToMuMu_M125.root",
    },
    1: {
        #"data": "data.root",
        #"dy_inc": "DY_105To160.root",
        #"dy_inc_fail": "DY_105To160_Inc_failvbffilter.root",
        #"dy_vbffilter": "DY_105To160_Fil-VBF_passvbffilter.root",
        #"ewk_zjj": "EWK_LLJJ_M105To160.root",
        #"ewk_zjj_pythia": "EWK_2Mu2J_105to160_pythia.root",
        #"TTTo2L2Nu": "TTTo2L2Nu.root",
        #"vbf_hmm": "VBFHToMuMu_M125_ggHUnc.root",
        #"ggh_hmm": "GluGluHToMuMu_M125_ggHUnc.root",
        #"WWTo2L2Nu": "WWTo2L2Nu.root",
        #"WWW": "WWW.root",
        #"WWZ": "WWZ.root",
        #"WZTo2L2Q": "WZTo2L2Q.root",
        #"WZTo3LNu": "WZTo3LNu.root",
        #"WZZ": "WZZ.root",
        #"ZZTo2L2Nu": "ZZTo2L2Nu.root",
        #"ZZTo2L2Q": "ZZTo2L2Q.root",
        #"ZZTo4L": "ZZTo4L.root",
        #"ZZZ": "ZZZ.root",
        #"ST_tW_antitop": "ST_tW_antitop.root",
        #"ST_tW_top": "ST_tW_top.root",
        #"vh_hmm": "VHToMuMu_M125.root",
        #"tth_hmm": "TTHto2Mu_M-125.root",
        #"rare_hmm": "rareHToMuMu_M125.root",
        "vbf_hmm_herwig": "VBFHto2Mu_M-125_powheg_herwig_2022EE.root",
    },
    2: {
        #"data": "data.root",
        #"dy_inc": "DY_105To160.root",
        #"dy_inc_fail": "DY_105To160_Inc_failvbffilter.root",
        #"dy_vbffilter": "DY_105To160_Fil-VBF_passvbffilter.root",
        #"ewk_zjj": "EWK_LLJJ_M105To160.root",
        #"ewk_zjj_pythia": "EWK_2Mu2J_105to160_pythia.root",
        #"TTTo2L2Nu": "TTTo2L2Nu.root",
        #"vbf_hmm": "VBFHToMuMu_M125_ggHUnc.root",
        #"ggh_hmm": "GluGluHToMuMu_M125_ggHUnc.root",
        #"WWTo2L2Nu": "WWTo2L2Nu.root",
        #"WWW": "WWW.root",
        #"WWZ": "WWZ.root",
        #"WZTo2L2Q": "WZTo2L2Q.root",
        #"WZTo3LNu": "WZTo3LNu.root",
        #"WZZ": "WZZ.root",
        #"ZZTo2L2Nu": "ZZTo2L2Nu.root",
        #"ZZTo2L2Q": "ZZTo2L2Q.root",
        #"ZZTo4L": "ZZTo4L.root",
        #"ZZZ": "ZZZ.root",
        #"ST_tW_antitop": "ST_tW_antitop.root",
        #"ST_tW_top": "ST_tW_top.root",
        "vh_hmm": "VHToMuMu_M125.root",
        "tth_hmm": "TTHto2Mu_M-125.root",
        "rare_hmm": "rareHToMuMu_M125.root",
    },
    3: {
        #"data": "data.root",
        #"dy_inc": "DY_105To160.root",
        #"dy_inc_fail": "DY_105To160_Inc_failvbffilter.root",
        #"dy_vbffilter": "DY_105To160_Fil-VBF_passvbffilter.root",
        #"ewk_zjj": "EWK_LLJJ_M105To160.root",
        #"ewk_zjj_pythia": "EWK_2Mu2J_105to160_pythia.root",
        #"TTTo2L2Nu": "TTTo2L2Nu.root",
        #"vbf_hmm": "VBFHToMuMu_M125_ggHUnc.root",
        #"ggh_hmm": "GluGluHToMuMu_M125_ggHUnc.root",
        #"WWTo2L2Nu": "WWTo2L2Nu.root",
        #"WWW": "WWW.root",
        #"WWZ": "WWZ.root",
        #"WZTo2L2Q": "WZTo2L2Q.root",
        #"WZTo3LNu": "WZTo3LNu.root",
        #"WZZ": "WZZ.root",
        #"ZZTo2L2Nu": "ZZTo2L2Nu.root",
        #"ZZTo2L2Q": "ZZTo2L2Q.root",
        #"ZZTo4L": "ZZTo4L.root",
        #"ZZZ": "ZZZ.root",
        #"ST_tW_antitop": "ST_tW_antitop.root",
        #"ST_tW_top": "ST_tW_top.root",
        "vh_hmm": "VHToMuMu_M125.root",
        "tth_hmm": "TTHto2Mu_M-125.root",
        "rare_hmm": "rareHToMuMu_M125.root",
    },
    4: {
        #"data": "data.root",
        #"dy_inc": "DY_105To160.root",
        #"dy_inc_fail": "DY_105To160_Inc_failvbffilter.root",
        #"dy_vbffilter": "DY_105To160_Fil-VBF_passvbffilter.root",
        #"ewk_zjj": "EWK_LLJJ_M105To160.root",
        #"ewk_zjj_pythia": "EWK_2Mu2J_105to160_pythia.root",
        #"TTTo2L2Nu": "TTTo2L2Nu.root",
        #"vbf_hmm": "VBFHToMuMu_M125.root",
        #"ggh_hmm": "GluGluHToMuMu_M125.root",
        #"WWTo2L2Nu": "WWTo2L2Nu.root",
        #"WWW": "WWW.root",
        #"WWZ": "WWZ.root",
        #"WZTo2L2Q": "WZTo2L2Q.root",
        #"WZTo3LNu": "WZTo3LNu.root",
        #"WZZ": "WZZ.root",
        #"ZZTo2L2Nu": "ZZTo2L2Nu.root",
        #"ZZTo2L2Q": "ZZTo2L2Q.root",
        #"ZZTo4L": "ZZTo4L.root",
        #"ZZZ": "ZZZ.root",
        #"ST_tW_antitop": "ST_tW_antitop.root",
        #"ST_tW_top": "ST_tW_top.root",
        "vh_hmm": "VHToMuMu_M125.root",
        "tth_hmm": "TTHto2Mu_M-125.root",
        "rare_hmm": "rareHToMuMu_M125.root",
    },
    5: {
        #"data": "data.root",
        #"dy_inc": "DY_105To160.root",
        #"dy_inc_fail": "DY_105To160_Inc_failvbffilter.root",
        #"dy_vbffilter": "DY_105To160_Fil-VBF_passvbffilter.root",
        #"ewk_zjj": "EWK_LLJJ_M105To160.root",
        #"ewk_zjj_pythia": "EWK_2Mu2J_105to160_pythia.root",
        #"TTTo2L2Nu": "TTTo2L2Nu.root",
        #"vbf_hmm": "VBFHToMuMu_M125.root",
        #"ggh_hmm": "GluGluHToMuMu_M125.root",
        #"WWTo2L2Nu": "WWTo2L2Nu.root",
        #"WWW": "WWW.root",
        #"WWZ": "WWZ.root",
        #"WZTo2L2Q": "WZTo2L2Q.root",
        #"WZTo3LNu": "WZTo3LNu.root",
        #"WZZ": "WZZ.root",
        #"ZZTo2L2Nu": "ZZTo2L2Nu.root",
        #"ZZTo2L2Q": "ZZTo2L2Q.root",
        #"ZZTo4L": "ZZTo4L.root",
        #"ZZZ": "ZZZ.root",
        #"ST_tW_antitop": "ST_tW_antitop.root",
        #"ST_tW_top": "ST_tW_top.root",
        "vh_hmm": "VHToMuMu_M125.root",
        "tth_hmm": "TTHto2Mu_M-125.root",
        "rare_hmm": "rareHToMuMu_M125.root",
    },
}

# Keep all years defined above, but run only the selected year for now.
# Set one era here for a targeted application, or None to process all eras.
ONLY_YEAR = None
ONLY_YEAR = "2022EE"

if ONLY_YEAR is not None:
    keep_keys = [k for k, v in era_names.items() if v == ONLY_YEAR]
    path = {k: v for k, v in path.items() if k in keep_keys}
    era_names = {k: v for k, v in era_names.items() if k in keep_keys}
    tree_name_by_path = {k: v for k, v in tree_name_by_path.items() if k in keep_keys}
    samples_by_path = {k: v for k, v in samples_by_path.items() if k in keep_keys}


# ============================================================
# Event ID columns
# ============================================================

EVENT_ID_CANDIDATES = [
    ["run", "lumi", "event"],
    ["run", "luminosityBlock", "event"],
    ["run", "luminosityBlock", "evt"],
    ["run", "lumi", "evt"],
]


# ============================================================
# Custom objects for loading DA model
# ============================================================

@tf.custom_gradient
def gradient_reverse(x, lambd):
    def grad(dy):
        return -lambd * dy, None
    return tf.identity(x), grad


@tf.keras.utils.register_keras_serializable(package="VBF")
class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lambd = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def call(self, x, training=None):
        return gradient_reverse(x, self.lambd)

    def get_config(self):
        return super().get_config()


CUSTOM_OBJECTS = {
    "GradientReversalLayer": GradientReversalLayer,
    "gradient_reverse": gradient_reverse,
}


# ============================================================
# Setup
# ============================================================

def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    print("[INFO] Available GPUs:", gpus, flush=True)

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[INFO] GPU memory growth enabled.", flush=True)
        except RuntimeError as e:
            print("[WARNING] Could not set memory growth:", e, flush=True)


# ============================================================
# Feature handling
# ============================================================

def default_features():
    feature_branches = [
        "diMufsr_kit_BSC_mass",
        "diMu-mass_resolution_abs",
        "diMu-mass_resolution",
        "dijet_mass",
        "log_dijet_mass",
        "z_star",
        "R_pT",
        "delta_eta_jj",
        "SoftActivityJetNjets5",
        "min_delta_eta_dimu_jets",
        "diMufsr_kit_BSC_pt",
        "diMufsr_kit_BSC_eta",
        "log_diMufsr_kit_BSC_pt",
        "jet1_pt",
        "jet1_eta",
        "jet1_phi",
        "jet2_pt",
        "jet2_eta",
        "jet2_phi",
    ]
    year_features = [f"era_{era}" for era in era_names.values()]
    return feature_branches + year_features


def load_features(model_dir):
    features_json = os.path.join(model_dir, "features.json")

    if os.path.exists(features_json):
        with open(features_json) as f:
            cfg = json.load(f)

        if "all_feature_branches" in cfg:
            features = cfg["all_feature_branches"]
        elif "feature_branches" in cfg:
            features = cfg["feature_branches"]
        else:
            raise RuntimeError(
                "features.json exists but cannot find all_feature_branches or feature_branches."
            )

        print("[INFO] Loaded features from:", features_json, flush=True)
    else:
        cfg = {}
        features = default_features()
        print("[WARNING] No features.json found. Using default features.", flush=True)

    print("[INFO] Number of features:", len(features), flush=True)
    print("[INFO] Features:", features, flush=True)

    return cfg, features


def is_2022_or_2023_era(era):
    era_text = str(era)
    return era_text.startswith("2022") or era_text.startswith("2023")


def era_to_source_year(era):
    era_text = str(era)
    for year in (2022, 2023, 2024, 2025):
        if era_text.startswith(str(year)):
            return float(year)
    raise RuntimeError(f"Cannot derive source_year from era={era}")


def branch_candidates(branch, era):
    candidates = list(BRANCH_ALIASES.get(branch, [branch]))

    if branch in BRANCH_ALIASES:
        preferred_token = "_rc_BSC_" if is_2022_or_2023_era(era) else "_kit_BSC_"
        preferred = [candidate for candidate in candidates if preferred_token in candidate]
        candidates = preferred + [candidate for candidate in candidates if candidate not in preferred]

    return list(dict.fromkeys(candidates))


def resolve_branch(branch, available, era):
    for candidate in branch_candidates(branch, era):
        if candidate in available:
            return candidate
    return None


def apply_branch_aliases(df, branch_map, input_path, tree_name, era):
    out = df.copy()
    aliases_used = []

    for requested, actual in branch_map.items():
        if requested != actual:
            out[requested] = out[actual]
            aliases_used.append((requested, actual))

    if aliases_used:
        print(
            f"[INFO] Branch aliases used for era={era}, tree={tree_name}, "
            f"file={input_path}",
            flush=True,
        )
        for requested, actual in aliases_used:
            print(f"       {requested} <- {actual}", flush=True)

    return out


def ensure_era_features(df, features, era, source_year_for_features=None):
    df = df.copy()
    feature_year = source_year_for_features if source_year_for_features is not None else era

    source_year = era_to_source_year(feature_year)
    if source_year_for_features is not None or "source_year" not in df.columns:
        df["source_year"] = source_year
    else:
        values = pd.to_numeric(df["source_year"], errors="coerce")
        df["source_year"] = values.fillna(source_year).astype(np.float32)

    for feat in features:
        if feat.startswith("era_"):
            era_value = feat.replace("era_", "")
            df[feat] = 1.0 if str(feature_year) == era_value else 0.0

    for branch, default in OPTIONAL_BRANCH_DEFAULTS.items():
        if branch not in df.columns:
            df[branch] = default

    return df


# ============================================================
# Load model/scaler/DNN_t
# ============================================================

def find_model_path(model_dir, model_fold):
    if APPLY_TAG == "DA":
        candidates = [
            os.path.join(model_dir, "models", f"simple_DA_model_fold_{model_fold}.keras"),
            os.path.join(model_dir, "models", f"simple_DA_model_fold_{model_fold}.h5"),
        ]
    elif APPLY_TAG == "noDomain":
        candidates = [
            os.path.join(model_dir, "models", f"simple_noDomain_model_fold_{model_fold}.keras"),
            os.path.join(model_dir, "models", f"simple_noDomain_model_fold_{model_fold}.h5"),
        ]
    else:
        raise RuntimeError("APPLY_TAG must be 'DA' or 'noDomain'")

    for p in candidates:
        if os.path.exists(p):
            return p

    raise RuntimeError(
        f"Cannot find simple-DNN {APPLY_TAG} model for model fold {model_fold} in {model_dir}"
    )


def find_scaler_path(model_dir, model_fold):
    candidates = [
        os.path.join(model_dir, "models", f"scaler_fold_{model_fold}.pkl"),
        os.path.join(model_dir, "models", f"scaler_fold_{model_fold}.joblib"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    raise RuntimeError(f"Cannot find scaler for model fold {model_fold} in {model_dir}")


def find_tsf_path(model_dir, model_fold):
    candidates = [
        os.path.join(model_dir, "models", f"DNN_tsf_{APPLY_TAG}_fold_{model_fold}.joblib"),
        os.path.join(model_dir, "models", f"DNN_tsf_{APPLY_TAG}_fold_{model_fold}.pkl"),
        os.path.join(model_dir, "models", f"DNN_tsf_fold_{model_fold}.joblib"),
        os.path.join(model_dir, "models", f"DNN_tsf_fold_{model_fold}.pkl"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    return None


def load_pickle_or_joblib(path_in):
    try:
        return joblib.load(path_in)
    except Exception:
        with open(path_in, "rb") as f:
            return pickle.load(f)


def load_all_objects(model_dir):
    models = {}
    scalers = {}
    tsfs = {}

    for model_fold in range(1, N_FOLDS + 1):
        model_path = find_model_path(model_dir, model_fold)
        scaler_path = find_scaler_path(model_dir, model_fold)
        tsf_path = find_tsf_path(model_dir, model_fold)

        print(f"[INFO] Loading model fold {model_fold}: {model_path}", flush=True)
        models[model_fold] = tf.keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
            safe_mode=False,
        )

        print(f"[INFO] Loading scaler fold {model_fold}: {scaler_path}", flush=True)
        scalers[model_fold] = load_pickle_or_joblib(scaler_path)

        if tsf_path is not None:
            print(f"[INFO] Loading DNN_t transformer fold {model_fold}: {tsf_path}", flush=True)
            tsfs[model_fold] = load_pickle_or_joblib(tsf_path)
        else:
            print(f"[WARNING] No DNN_t transformer for model fold {model_fold}. dnn_t will be -999.", flush=True)
            tsfs[model_fold] = None

    return models, scalers, tsfs


# ============================================================
# Fold utilities
# ============================================================

def get_event_id_cols(df):
    for cols in EVENT_ID_CANDIDATES:
        if all(c in df.columns for c in cols):
            return cols
    return None


def stable_hash_int(values, mod):
    key = "|".join(str(v) for v in values)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    val = int(digest[:8], 16)
    return val % mod


def stable_hash_to_event_fold(values, n_folds=4):
    return stable_hash_int(values, n_folds) + 1


def model_fold_from_event_fold(event_fold):
    event_fold = int(event_fold)

    if event_fold not in APPLICATION_MODEL_FOR_EVENT_FOLD:
        raise RuntimeError(f"Unknown event fold {event_fold}")

    return int(APPLICATION_MODEL_FOR_EVENT_FOLD[event_fold])


def add_model_fold_id(df):
    df = df.copy()
    df["event_fold_id"] = df["event_fold_id"].astype(np.int32)
    df["model_fold_id"] = np.array(
        [model_fold_from_event_fold(x) for x in df["event_fold_id"].values],
        dtype=np.int32,
    )
    return df


def assign_random_hash_folds(df, sample_short, era, fold_method):
    """
    Deterministic random/hash fallback.
    Used when task/domain pkl does not contain run/lumi/event.
    """
    df = df.copy()
    id_cols = get_event_id_cols(df)

    if id_cols is not None:
        print(f"[INFO] Assign event folds by deterministic hash using {id_cols}", flush=True)
        event_fold_ids = [
            stable_hash_to_event_fold(
                (RANDOM_FOLD_SEED_TAG, era, sample_short) + tuple(vals),
                N_FOLDS,
            )
            for vals in df[id_cols].itertuples(index=False, name=None)
        ]
    else:
        print("[WARNING] No event id columns in ROOT. Use row-index deterministic hash.", flush=True)
        event_fold_ids = [
            stable_hash_to_event_fold(
                (RANDOM_FOLD_SEED_TAG, era, sample_short, i),
                N_FOLDS,
            )
            for i in range(len(df))
        ]

    df["event_fold_id"] = np.array(event_fold_ids, dtype=np.int32)
    df["fold_method"] = np.full(len(df), fold_method, dtype=np.int32)

    df = add_model_fold_id(df)

    return df


# ============================================================
# Fold map loading and exact matching
# ============================================================

def load_one_fold_map(model_dir, filename, map_name):
    p = os.path.join(model_dir, filename)

    if not os.path.exists(p):
        print(f"[WARNING] {filename} not found.", flush=True)
        return None

    print(f"[INFO] Loading {map_name} fold map: {p}", flush=True)
    fold_df = pd.read_pickle(p)

    print(f"[INFO] {map_name} fold map loaded.", flush=True)
    print("[INFO] rows:", len(fold_df), flush=True)
    print("[INFO] columns:", list(fold_df.columns), flush=True)

    if "fold_id" not in fold_df.columns:
        print(f"[WARNING] {filename} has no fold_id column.", flush=True)
        return None

    id_cols = get_event_id_cols(fold_df)

    if id_cols is None:
        print(f"[WARNING] {filename} does NOT have run/lumi/event columns.", flush=True)
        print(f"[WARNING] Exact out-of-fold matching is impossible for {map_name} map.", flush=True)

        return {
            "df": fold_df[["fold_id"]].copy(),
            "id_cols": None,
            "has_event_id": False,
            "has_source_file": "source_file" in fold_df.columns,
            "has_era": "era" in fold_df.columns,
            "name": map_name,
            "filename": filename,
        }

    keep_cols = id_cols + ["fold_id"]

    if "source_file" in fold_df.columns:
        keep_cols = ["source_file"] + keep_cols

    if "era" in fold_df.columns:
        keep_cols = ["era"] + keep_cols

    fold_df = fold_df[keep_cols].drop_duplicates()

    print(f"[INFO] {map_name} map supports EXACT event-id matching.", flush=True)
    print(f"[INFO] {map_name} id columns:", id_cols, flush=True)
    print(f"[INFO] {map_name} keep columns:", keep_cols, flush=True)
    print(f"[INFO] {map_name} size after drop_duplicates:", len(fold_df), flush=True)

    return {
        "df": fold_df,
        "id_cols": id_cols,
        "has_event_id": True,
        "has_source_file": "source_file" in fold_df.columns,
        "has_era": "era" in fold_df.columns,
        "name": map_name,
        "filename": filename,
    }


def load_training_fold_maps(model_dir):
    task_map = load_one_fold_map(
        model_dir,
        filename="task_df_with_folds.pkl",
        map_name="task",
    )

    domain_map = load_one_fold_map(
        model_dir,
        filename="domain_df_with_folds.pkl",
        map_name="domain",
    )

    return task_map, domain_map


def exact_match_fold_map(
    df,
    fold_map,
    source_file,
    era,
    fold_method_base,
    strict_keys=False,
):
    """
    Exact event-id matching using run/lumi/event if available in pkl.
    """
    if fold_map is None:
        return df, False

    if not fold_map.get("has_event_id", False):
        return df, False

    id_cols = fold_map["id_cols"]

    if not all(c in df.columns for c in id_cols):
        print(
            f"[WARNING] ROOT input does not contain {fold_map['name']} map id columns: {id_cols}",
            flush=True,
        )
        return df, False

    def _try_merge(use_source_file=True, use_era=True):
        left = df[id_cols].copy()
        left["_rowid_"] = np.arange(len(df))

        merge_cols = id_cols.copy()

        if use_source_file and fold_map["has_source_file"]:
            left["source_file"] = source_file
            merge_cols = ["source_file"] + merge_cols

        if use_era and fold_map["has_era"]:
            left["era"] = era
            merge_cols = ["era"] + merge_cols

        right_cols = merge_cols + ["fold_id"]

        for c in right_cols:
            if c not in fold_map["df"].columns:
                return None

        right = fold_map["df"][right_cols].drop_duplicates()
        duplicate_key_mask = right.duplicated(subset=merge_cols, keep=False)
        if duplicate_key_mask.any():
            n_duplicate_rows = int(duplicate_key_mask.sum())
            n_duplicate_keys = int(right.loc[duplicate_key_mask, merge_cols].drop_duplicates().shape[0])
            n_conflicting_keys = int(
                (right.loc[duplicate_key_mask].groupby(merge_cols)["fold_id"].nunique() > 1).sum()
            )
            print(
                f"[WARNING] {fold_map['name']} fold map has {n_duplicate_rows} duplicate rows "
                f"for {n_duplicate_keys} merge keys using {merge_cols}; "
                f"{n_conflicting_keys} keys have conflicting fold_id. Keeping the first fold_id.",
                flush=True,
            )
            right = right.drop_duplicates(subset=merge_cols, keep="first")

        merged = left.merge(right, on=merge_cols, how="left")
        merged = merged.sort_values("_rowid_")
        if len(merged) != len(df):
            print(
                f"[WARNING] {fold_map['name']} exact merge using {merge_cols} changed row count "
                f"from {len(df)} to {len(merged)}. Skip this matching mode.",
                flush=True,
            )
            return None

        n_match = int(merged["fold_id"].notna().sum())

        return merged, n_match, merge_cols

    # Try source_file + era + event id.
    if fold_map["has_source_file"]:
        trial = _try_merge(use_source_file=True, use_era=True)
        if trial is not None:
            merged, n_match, merge_cols = trial
            print(
                f"[INFO] {fold_map['name']} exact match using {merge_cols}: matched {n_match}/{len(df)}",
                flush=True,
            )
            if n_match > 0:
                out = df.copy()
                out["event_fold_id"] = merged["fold_id"].values
                out["fold_method"] = fold_method_base + 0
                return out, True
        if strict_keys:
            return df, False

    # Try era + event id.
    trial = _try_merge(use_source_file=False, use_era=True)
    if trial is not None:
        merged, n_match, merge_cols = trial
        print(
            f"[INFO] {fold_map['name']} exact match using {merge_cols}: matched {n_match}/{len(df)}",
            flush=True,
        )
        if n_match > 0:
            out = df.copy()
            out["event_fold_id"] = merged["fold_id"].values
            out["fold_method"] = fold_method_base + 1
            return out, True
    if strict_keys and fold_map["has_era"]:
        return df, False

    # Try event id only.
    trial = _try_merge(use_source_file=False, use_era=False)
    if trial is not None:
        merged, n_match, merge_cols = trial
        print(
            f"[INFO] {fold_map['name']} exact match using {merge_cols}: matched {n_match}/{len(df)}",
            flush=True,
        )
        if n_match > 0:
            out = df.copy()
            out["event_fold_id"] = merged["fold_id"].values
            out["fold_method"] = fold_method_base + 2
            return out, True

    return df, False


def is_data_sample(sample_short, filename):
    s = sample_short.lower()
    f = filename.lower()

    if s == "data" or s.startswith("data"):
        return True

    if "data" in f and "dy" not in f:
        return True

    return False


def is_task_training_sample(sample_short):
    return sample_short in TASK_TRAINING_SAMPLES


def is_domain_training_sample(sample_short):
    return sample_short in DOMAIN_TRAINING_SAMPLES


#def assign_application_folds(
#    df,
#    sample_short,
#    source_file,
#    era,
#    is_data,
#    task_map,
#    domain_map,
#):
#    """
#    Decide event_fold_id and model_fold_id.
#
#    SR:
#      if sample used in task training and task map has event IDs:
#          exact task fold
#      else:
#          deterministic random/hash fallback if enabled
#
#    SB:
#      if sample used in domain training and domain map has event IDs:
#          exact domain fold
#      else:
#          deterministic random/hash fallback if enabled
#    """
#    df = df.copy()
#
#    if MASS_BRANCH in df.columns:
#        mass_values = df[MASS_BRANCH].astype(float).values
#        is_sb_event = (mass_values < SR_LOW) | (mass_values > SR_HIGH)
#    else:
#        print("[WARNING] Mass branch not found when assigning folds. Treat all events as SB.", flush=True)
#        is_sb_event = np.ones(len(df), dtype=bool)
#
#    is_sr_event = ~is_sb_event
#
#    df["event_fold_id"] = np.nan
#    df["model_fold_id"] = np.nan
#    df["fold_method"] = -1
#
#    print("[INFO] Fold assignment summary:", flush=True)
#    print(f"       sample_short = {sample_short}", flush=True)
#    print(f"       is_data = {is_data}", flush=True)
#    print(f"       total events = {len(df)}", flush=True)
#    print(f"       SR events = {int(np.sum(is_sr_event))}", flush=True)
#    print(f"       SB events = {int(np.sum(is_sb_event))}", flush=True)
#    print(f"       task-training sample = {is_task_training_sample(sample_short)}", flush=True)
#    print(f"       domain-training sample = {is_domain_training_sample(sample_short)}", flush=True)
#
#    # ========================================================
#    # SR events
#    # ========================================================
#    if np.sum(is_sr_event) > 0:
#        df_sr = df.loc[is_sr_event].copy()
#
#        use_exact_task = (
#            (not is_data)
#            and is_task_training_sample(sample_short)
#            and task_map is not None
#            and task_map.get("has_event_id", False)
#        )
#
#        if use_exact_task:
#            df_sr_mapped, used = exact_match_fold_map(
#                df_sr,
#                task_map,
#                source_file=source_file,
#                era=era,
#                fold_method_base=100,
#            )
#
#            if used:
#                missing = pd.isna(df_sr_mapped["event_fold_id"])
#                n_missing = int(missing.sum())
#
#                if n_missing > 0:
#                    raise RuntimeError(
#                        f"SR task sample {sample_short}: exact fold-map partial match failed. "
#                        f"Missing {n_missing}/{len(df_sr_mapped)}."
#                    )
#
#                df.loc[df_sr_mapped.index, "event_fold_id"] = df_sr_mapped["event_fold_id"].values
#                df.loc[df_sr_mapped.index, "model_fold_id"] = df_sr_mapped["model_fold_id"].values
#                df.loc[df_sr_mapped.index, "fold_method"] = df_sr_mapped["fold_method"].values
#            else:
#                raise RuntimeError(
#                    f"SR task sample {sample_short}: task map has event IDs but no events matched."
#                )
#
#        else:
#            if (
#                (not is_data)
#                and is_task_training_sample(sample_short)
#                and task_map is not None
#                and not task_map.get("has_event_id", False)
#            ):
#                print(
#                    f"[WARNING] task_df_with_folds.pkl has no event IDs. "
#                    f"Using random/hash fallback for SR task sample {sample_short}.",
#                    flush=True,
#                )
#
#            if not USE_RANDOM_HASH_FOLD_FOR_APPLICATION:
#                raise RuntimeError(
#                    "Need random/hash fallback for SR events, but "
#                    "USE_RANDOM_HASH_FOLD_FOR_APPLICATION=False."
#                )
#
#            fallback = assign_random_hash_folds(
#                df_sr,
#                sample_short=sample_short,
#                era=era,
#                fold_method=190,
#            )
#
#            df.loc[fallback.index, "event_fold_id"] = fallback["event_fold_id"].values
#            df.loc[fallback.index, "model_fold_id"] = fallback["model_fold_id"].values
#            df.loc[fallback.index, "fold_method"] = fallback["fold_method"].values
#
#    # ========================================================
#    # SB events
#    # ========================================================
#    if np.sum(is_sb_event) > 0:
#        df_sb = df.loc[is_sb_event].copy()
#
#        use_exact_domain = (
#            is_domain_training_sample(sample_short)
#            and domain_map is not None
#            and domain_map.get("has_event_id", False)
#        )
#
#        if use_exact_domain:
#            df_sb_mapped, used = exact_match_fold_map(
#                df_sb,
#                domain_map,
#                source_file=source_file,
#                era=era,
#                fold_method_base=200,
#            )
#
#            if used:
#                missing = pd.isna(df_sb_mapped["event_fold_id"])
#                n_missing = int(missing.sum())
#
#                if n_missing > 0:
#                    raise RuntimeError(
#                        f"SB domain sample {sample_short}: exact fold-map partial match failed. "
#                        f"Missing {n_missing}/{len(df_sb_mapped)}."
#                    )
#
#                df.loc[df_sb_mapped.index, "event_fold_id"] = df_sb_mapped["event_fold_id"].values
#                df.loc[df_sb_mapped.index, "model_fold_id"] = df_sb_mapped["model_fold_id"].values
#                df.loc[df_sb_mapped.index, "fold_method"] = df_sb_mapped["fold_method"].values
#            else:
#                raise RuntimeError(
#                    f"SB domain sample {sample_short}: domain map has event IDs but no events matched."
#                )
#
#        else:
#            if (
#                is_domain_training_sample(sample_short)
#                and domain_map is not None
#                and not domain_map.get("has_event_id", False)
#            ):
#                print(
#                    f"[WARNING] domain_df_with_folds.pkl has no event IDs. "
#                    f"Using random/hash fallback for SB domain sample {sample_short}.",
#                    flush=True,
#                )
#
#            if not USE_RANDOM_HASH_FOLD_FOR_APPLICATION:
#                raise RuntimeError(
#                    "Need random/hash fallback for SB events, but "
#                    "USE_RANDOM_HASH_FOLD_FOR_APPLICATION=False."
#                )
#
#            fallback = assign_random_hash_folds(
#                df_sb,
#                sample_short=sample_short,
#                era=era,
#                fold_method=290,
#            )
#
#            df.loc[fallback.index, "event_fold_id"] = fallback["event_fold_id"].values
#            df.loc[fallback.index, "model_fold_id"] = fallback["model_fold_id"].values
#            df.loc[fallback.index, "fold_method"] = fallback["fold_method"].values
#
#    if df["event_fold_id"].isna().any():
#        n_bad = int(df["event_fold_id"].isna().sum())
#        raise RuntimeError(f"Internal error: {n_bad} events still have NaN event_fold_id.")
#
#    if df["model_fold_id"].isna().any():
#        n_bad = int(df["model_fold_id"].isna().sum())
#        raise RuntimeError(f"Internal error: {n_bad} events still have NaN model_fold_id.")
#
#    df["event_fold_id"] = df["event_fold_id"].astype(np.int32)
#    df["model_fold_id"] = df["model_fold_id"].astype(np.int32)
#    df["fold_method"] = df["fold_method"].astype(np.int32)
#
#    return df

def assign_application_folds(
    df,
    sample_short,
    source_file,
    era,
    is_data,
    task_map,
    domain_map,
    allow_hash_for_training=False,
    use_exact_fold_maps=True,
):
    df = df.copy()

    if MASS_BRANCH in df.columns:
        mass_values = df[MASS_BRANCH].astype(float).values
        is_sb_event = (mass_values < SR_LOW) | (mass_values > SR_HIGH)
    else:
        print("[WARNING] Mass branch not found when assigning folds. Treat all events as SB.", flush=True)
        is_sb_event = np.ones(len(df), dtype=bool)

    is_sr_event = ~is_sb_event

    df["event_fold_id"] = np.nan
    df["model_fold_id"] = np.nan
    df["fold_method"] = -1

    is_task_train_sample = is_task_training_sample(sample_short)
    is_domain_train_sample = is_domain_training_sample(sample_short)

    print("[INFO] Fold assignment summary:", flush=True)
    print(f"       sample_short = {sample_short}", flush=True)
    print(f"       is_data = {is_data}", flush=True)
    print(f"       total events = {len(df)}", flush=True)
    print(f"       SR events = {int(np.sum(is_sr_event))}", flush=True)
    print(f"       SB events = {int(np.sum(is_sb_event))}", flush=True)
    print(f"       task-training sample = {is_task_train_sample}", flush=True)
    print(f"       domain-training sample = {is_domain_train_sample}", flush=True)

    def fill_from_hash(df_part, fold_method, reason):
        print(f"[INFO] {reason} Use deterministic hash fold assignment.", flush=True)
        if not USE_RANDOM_HASH_FOLD_FOR_APPLICATION:
            raise RuntimeError(
                "Need hash fallback for application, but "
                "USE_RANDOM_HASH_FOLD_FOR_APPLICATION=False."
            )
        fallback = assign_random_hash_folds(
            df_part,
            sample_short=sample_short,
            era=era,
            fold_method=fold_method,
        )
        df.loc[fallback.index, "event_fold_id"] = fallback["event_fold_id"].values
        df.loc[fallback.index, "model_fold_id"] = fallback["model_fold_id"].values
        df.loc[fallback.index, "fold_method"] = fallback["fold_method"].values

    def small_unmatched_tail_is_allowed(n_missing, population_size, label):
        population_size = max(int(population_size), 1)
        fraction = float(n_missing) / float(population_size)
        allowed = (
            n_missing <= ALWAYS_ALLOW_UNMATCHED_TRAINING_EVENTS_FOR_HASH
            or (
                n_missing <= MAX_UNMATCHED_TRAINING_EVENTS_FOR_HASH
                and fraction <= MAX_UNMATCHED_TRAINING_FRACTION_FOR_HASH
            )
        )
        if allowed:
            print(
                f"[WARNING] {label} sample {sample_short}: allow deterministic hash "
                f"for tiny reskim mismatch {n_missing}/{population_size} "
                f"({fraction:.3e}).",
                flush=True,
            )
        return allowed

    def fill_from_exact_or_hash(
        df_part,
        fold_map,
        fold_method_base,
        hash_method,
        label,
        population_size,
    ):
        mapped, used = exact_match_fold_map(
            df_part,
            fold_map,
            source_file=source_file,
            era=era,
            fold_method_base=fold_method_base,
            strict_keys=use_exact_fold_maps,
        )

        if used:
            exact_mask = mapped["event_fold_id"].notna()

            if exact_mask.any():
                exact = mapped.loc[exact_mask].copy()
                exact["event_fold_id"] = exact["event_fold_id"].astype(np.int32)
                exact["model_fold_id"] = np.array(
                    [model_fold_from_event_fold(x) for x in exact["event_fold_id"].values],
                    dtype=np.int32,
                )
                exact["fold_method"] = exact["fold_method"].astype(np.int32)

                df.loc[exact.index, "event_fold_id"] = exact["event_fold_id"].values
                df.loc[exact.index, "model_fold_id"] = exact["model_fold_id"].values
                df.loc[exact.index, "fold_method"] = exact["fold_method"].values

            missing_mask = mapped["event_fold_id"].isna()
            n_missing = int(missing_mask.sum())
            if n_missing == 0:
                return

            if (
                not allow_hash_for_training
                and not small_unmatched_tail_is_allowed(
                    n_missing,
                    population_size,
                    label,
                )
            ):
                raise RuntimeError(
                    f"{label} sample {sample_short}: exact fold matching failed for "
                    f"{n_missing}/{population_size} events. The mismatch exceeds "
                    f"ALWAYS_ALLOW_UNMATCHED_TRAINING_EVENTS_FOR_HASH="
                    f"{ALWAYS_ALLOW_UNMATCHED_TRAINING_EVENTS_FOR_HASH}, "
                    f"MAX_UNMATCHED_TRAINING_EVENTS_FOR_HASH="
                    f"{MAX_UNMATCHED_TRAINING_EVENTS_FOR_HASH} or "
                    f"MAX_UNMATCHED_TRAINING_FRACTION_FOR_HASH="
                    f"{MAX_UNMATCHED_TRAINING_FRACTION_FOR_HASH}. "
                    f"source_file={source_file}, era={era}."
                )

            fill_from_hash(
                df_part.loc[missing_mask].copy(),
                hash_method,
                f"[WARNING] {label} sample {sample_short}: exact fold matching found "
                f"{len(df_part) - n_missing}/{len(df_part)} events; hashing the remaining "
                f"{n_missing}.",
            )
            return

        n_missing = len(df_part)
        if (
            not allow_hash_for_training
            and not small_unmatched_tail_is_allowed(
                n_missing,
                population_size,
                label,
            )
        ):
            raise RuntimeError(
                f"{label} sample {sample_short}: exact fold matching failed "
                f"for {n_missing}/{population_size} events. source_file={source_file}, era={era}."
            )

        fill_from_hash(
            df_part,
            hash_method,
            f"[WARNING] {label} sample {sample_short}: exact fold matching failed "
            f"for {n_missing}/{len(df_part)} events.",
        )

    def fill_from_exact_map_only(df_part, fold_map, fold_method_base, label):
        """Assign every event found in a training map before using shifted regions."""
        remaining_indices = df_part.index[
            df.loc[df_part.index, "event_fold_id"].isna()
        ]
        n_matched_total = 0

        # A partial source_file/era match can leave rows that match a less
        # specific key. Retry the unresolved subset so all matching modes are
        # exhausted before declaring an event absent from the training map.
        for _ in range(3):
            if len(remaining_indices) == 0:
                break

            current = df.loc[remaining_indices].copy()
            mapped, used = exact_match_fold_map(
                current,
                fold_map,
                source_file=source_file,
                era=era,
                fold_method_base=fold_method_base,
                strict_keys=True,
            )
            if not used:
                break

            exact_mask = mapped["event_fold_id"].notna()
            if not exact_mask.any():
                break

            exact = mapped.loc[exact_mask].copy()
            exact["event_fold_id"] = exact["event_fold_id"].astype(np.int32)
            exact["model_fold_id"] = np.array(
                [model_fold_from_event_fold(x) for x in exact["event_fold_id"].values],
                dtype=np.int32,
            )
            exact["fold_method"] = exact["fold_method"].astype(np.int32)

            df.loc[exact.index, "event_fold_id"] = exact["event_fold_id"].values
            df.loc[exact.index, "model_fold_id"] = exact["model_fold_id"].values
            df.loc[exact.index, "fold_method"] = exact["fold_method"].values
            n_matched_total += len(exact)

            remaining_indices = remaining_indices.difference(exact.index, sort=False)

        print(
            f"[INFO] {label}: pre-assigned exact out-of-fold model for "
            f"{n_matched_total}/{len(df_part)} events.",
            flush=True,
        )

    # Match training membership before looking at the current SR/SB region.
    # This keeps the original out-of-fold model when a systematic variation
    # moves an event across the mass-window boundary.
    exact_maps = []
    if (not is_data) and is_task_train_sample:
        exact_maps.append(("task training map", task_map, 100))
    if is_domain_train_sample:
        exact_maps.append(("domain training map", domain_map, 200))

    if use_exact_fold_maps:
        for label, fold_map, method_base in exact_maps:
            if fold_map is None or not fold_map.get("has_event_id", False):
                raise RuntimeError(
                    f"{label} is unavailable for era={era}, sample={sample_short}. "
                    "Cannot guarantee leakage-free out-of-fold application."
                )

            missing_id_cols = [c for c in fold_map["id_cols"] if c not in df.columns]
            if missing_id_cols:
                raise RuntimeError(
                    f"ROOT tree is missing {missing_id_cols} required by {label}. "
                    "Cannot guarantee leakage-free out-of-fold application."
                )

            unresolved = df.loc[df["event_fold_id"].isna()].copy()
            fill_from_exact_map_only(
                unresolved,
                fold_map=fold_map,
                fold_method_base=method_base,
                label=label,
            )

    unassigned_sr = is_sr_event & df["event_fold_id"].isna().to_numpy()
    if np.sum(unassigned_sr) > 0:
        df_sr = df.loc[unassigned_sr].copy()
        use_exact_task = (
            use_exact_fold_maps
            and (not is_data)
            and is_task_train_sample
            and task_map is not None
            and task_map.get("has_event_id", False)
        )

        if use_exact_task:
            fill_from_exact_or_hash(
                df_sr,
                task_map,
                fold_method_base=100,
                hash_method=191,
                label="SR task",
                population_size=int(np.sum(is_sr_event)),
            )
        else:
            if (not is_data) and is_task_train_sample:
                if not allow_hash_for_training:
                    raise RuntimeError(
                        f"SR task sample {sample_short}: exact task fold matching is unavailable. "
                        "Cannot avoid training leakage for nominal application."
                    )
                fill_from_hash(
                        df_sr,
                    fold_method=193,
                    reason=f"[WARNING] SR task sample {sample_short}: exact task map unavailable.",
                )
            else:
                fill_from_hash(
                    df_sr,
                    fold_method=190,
                    reason=f"SR events for {sample_short} are not task-training events.",
                )

    unassigned_sb = is_sb_event & df["event_fold_id"].isna().to_numpy()
    if np.sum(unassigned_sb) > 0:
        df_sb = df.loc[unassigned_sb].copy()
        use_exact_domain = (
            use_exact_fold_maps
            and is_domain_train_sample
            and domain_map is not None
            and domain_map.get("has_event_id", False)
        )

        if use_exact_domain:
            fill_from_exact_or_hash(
                df_sb,
                domain_map,
                fold_method_base=200,
                hash_method=291,
                label="SB domain",
                population_size=int(np.sum(is_sb_event)),
            )
        else:
            if is_domain_train_sample:
                if not allow_hash_for_training:
                    raise RuntimeError(
                        f"SB domain sample {sample_short}: exact domain fold matching is unavailable. "
                        "Cannot avoid domain-training leakage for nominal application."
                    )
                fill_from_hash(
                        df_sb,
                    fold_method=293,
                    reason=f"[WARNING] SB domain sample {sample_short}: exact domain map unavailable.",
                )
            else:
                fill_from_hash(
                    df_sb,
                    fold_method=290,
                    reason=f"SB events for {sample_short} are not domain-training events.",
                )

    if df["event_fold_id"].isna().any():
        n_bad = int(df["event_fold_id"].isna().sum())
        raise RuntimeError(f"Internal error: {n_bad} events still have NaN event_fold_id.")

    if df["model_fold_id"].isna().any():
        n_bad = int(df["model_fold_id"].isna().sum())
        raise RuntimeError(f"Internal error: {n_bad} events still have NaN model_fold_id.")

    df["event_fold_id"] = df["event_fold_id"].astype(np.int32)
    df["model_fold_id"] = df["model_fold_id"].astype(np.int32)
    df["fold_method"] = df["fold_method"].astype(np.int32)

    return df


# ============================================================
# DNN input preparation and prediction
# ============================================================

def prepare_features_for_dnn(df, features):
    df_work = df.copy()

    if MASS_BRANCH not in df_work.columns:
        raise RuntimeError(f"Missing mass branch {MASS_BRANCH}")

    original_mass = df_work[MASS_BRANCH].astype(float).values
    is_sb = (original_mass < SR_LOW) | (original_mass > SR_HIGH)

    mass_original = original_mass.astype(np.float32)
    mass_used = original_mass.copy().astype(np.float32)
    mass_fixed = is_sb.astype(np.int32)

    if FIX_SB_MASS_TO_125:
        mass_used[is_sb] = FIXED_MASS_VALUE
        df_work.loc[is_sb, MASS_BRANCH] = FIXED_MASS_VALUE

    for feat in features:
        if feat not in df_work.columns:
            raise RuntimeError(f"Missing model input feature: {feat}")

    X = df_work[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(-1.0)

    mass_info = pd.DataFrame({
        "dnn_mass_original": mass_original,
        "dnn_mass_used": mass_used,
        "dnn_mass_was_fixed": mass_fixed,
    }, index=df.index)

    return X.values.astype(np.float32), mass_info


def predict_simple_dnn(model, X_scaled):
    pred = model.predict(
        {"features": X_scaled.astype(np.float32)},
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    if isinstance(pred, list):
        outputs = [p.reshape(-1).astype(np.float32) for p in pred]
    else:
        outputs = [pred.reshape(-1).astype(np.float32)]

    n = len(X_scaled)

    out = {
        "dnn_score": np.full(n, -999.0, dtype=np.float32),
        "dnn_domain_score": np.full(n, -999.0, dtype=np.float32),
    }

    if len(outputs) >= 1:
        out["dnn_score"] = outputs[0]

    if len(outputs) >= 2:
        out["dnn_domain_score"] = outputs[1]

    return out


def apply_models_to_df(df, features, models, scalers, tsfs):
    df_out = df.copy()
    n = len(df_out)

    new_cols = [
        "dnn_score",
        "dnn_t",
        "dnn_domain_score",
        "dnn_mass_original",
        "dnn_mass_used",
        "dnn_mass_was_fixed",
    ]

    for col in new_cols:
        if col == "dnn_mass_was_fixed":
            df_out[col] = np.full(n, -1, dtype=np.int32)
        else:
            df_out[col] = np.full(n, -999.0, dtype=np.float32)

    for model_fold in range(1, N_FOLDS + 1):
        mask = df_out["model_fold_id"].values == model_fold

        if np.sum(mask) == 0:
            continue

        print(
            f"[INFO] Applying model fold {model_fold} to {np.sum(mask)} events",
            flush=True,
        )

        df_fold = df_out.loc[mask].copy()
        X_raw, mass_info = prepare_features_for_dnn(df_fold, features)
        X_scaled = scalers[model_fold].transform(X_raw).astype(np.float32)

        pred = predict_simple_dnn(models[model_fold], X_scaled)

        for key, arr in pred.items():
            df_out.loc[mask, key] = arr

        if tsfs[model_fold] is not None:
            dnn_t = tsfs[model_fold].transform(
                pred["dnn_score"].reshape(-1, 1)
            ).reshape(-1)
            df_out.loc[mask, "dnn_t"] = dnn_t.astype(np.float32)

        df_out.loc[mask, "dnn_mass_original"] = mass_info["dnn_mass_original"].values.astype(np.float32)
        df_out.loc[mask, "dnn_mass_used"] = mass_info["dnn_mass_used"].values.astype(np.float32)
        df_out.loc[mask, "dnn_mass_was_fixed"] = mass_info["dnn_mass_was_fixed"].values.astype(np.int32)

    return df_out


# ============================================================
# ROOT IO
# ============================================================

def get_available_branches(input_path, tree_name):
    with uproot.open(input_path) as f:
        if tree_name not in f:
            raise RuntimeError(
                f"Tree {tree_name} not found in {input_path}. Available: {list(f.keys())}"
            )
        return list(f[tree_name].keys())


def clean_root_key(key):
    return str(key).split(";")[0]


def discover_trees(input_path, nominal_tree, tree_mode="all"):
    with uproot.open(input_path) as f:
        tree_names = []
        for key, obj in f.items():
            name = clean_root_key(key)
            if hasattr(obj, "arrays"):
                tree_names.append(name)

    tree_names = sorted(set(tree_names))

    if tree_mode == "nominal":
        selected = [nominal_tree] if nominal_tree in tree_names else []
    else:
        selected = [
            t for t in tree_names
            if t == nominal_tree or t.startswith(f"{nominal_tree}__")
        ]

    if not selected:
        raise RuntimeError(
            f"No trees selected in {input_path}. nominal_tree={nominal_tree}, "
            f"tree_mode={tree_mode}. Available trees: {tree_names}"
        )

    return selected


def is_systematic_tree(tree_name, nominal_tree):
    return tree_name.startswith(f"{nominal_tree}__")


def build_read_branches(input_path, tree_name, features, era):
    available = set(get_available_branches(input_path, tree_name))

    read = set()
    branch_map = {}
    missing_features = []

    for feat in features:
        if feat.startswith("era_") or feat == "source_year":
            continue

        actual = resolve_branch(feat, available, era)
        if actual is None:
            if feat not in OPTIONAL_BRANCH_DEFAULTS:
                missing_features.append(feat)
        else:
            read.add(actual)
            branch_map[feat] = actual

    keep_extra = [
        WEIGHT_BRANCH,
        "eventWeight",
        "nominal_wgt",
        MASS_BRANCH,
        "diMufsr_rc_mass",
        "diMufsr_rc_pt",
        CATEGORY_BRANCH,
        "dijet_mass",
        "log_dijet_mass",
        "delta_eta_jj",
        "z_star",
        "R_pT",
        "SoftActivityJetNjets5",
        "min_delta_eta_dimu_jets",
        "jet1_pt",
        "jet1_eta",
        "jet1_phi",
        "jet1_mass",
        "jet2_pt",
        "jet2_eta",
        "jet2_phi",
        "jet2_mass",
        "njets",
        "nmuons",
        "source_year",
        "trg_single_mu24",
        "run",
        "lumi",
        "event",
        "genvbffilter_flag",
        "n_jets_matched_genjet",
        "iso_MuonEffup",
        "iso_MuonEffdown",
        "id_MuonEffup",
        "id_MuonEffdown",
        "PDF_uncertainty_up",
        "PDF_uncertainty_down",
        "qcd_unc_up",
        "qcd_unc_down",
    ]

    for b in keep_extra:
        actual = resolve_branch(b, available, era)
        if actual is not None:
            read.add(actual)
            branch_map.setdefault(b, actual)

    if missing_features:
        details = {
            feat: branch_candidates(feat, era)
            for feat in missing_features
        }
        raise RuntimeError(
            f"Missing required model features in {input_path}:{tree_name}: {details}"
        )

    return sorted(read), branch_map


def make_root_output_dict(df):
    out = {}

    for col in df.columns:
        if df[col].dtype == object:
            continue

        arr = df[col].values

        if arr.dtype == bool:
            arr = arr.astype(np.int8)

        out[col] = arr

    return out


def score_one_tree(
    input_path,
    tree_name,
    nominal_tree,
    sample_short,
    era,
    features,
    models,
    scalers,
    tsfs,
    task_map,
    domain_map,
):
    print("\n" + "=" * 100, flush=True)
    print("[INFO] Input :", input_path, flush=True)
    print("[INFO] Tree  :", tree_name, flush=True)
    print("[INFO] Sample:", sample_short, "Era:", era, flush=True)
    print("=" * 100, flush=True)

    if not os.path.exists(input_path):
        print("[WARNING] Input file does not exist. Skip:", input_path, flush=True)
        return None

    read_branches, branch_map = build_read_branches(
        input_path=input_path,
        tree_name=tree_name,
        features=features,
        era=era,
    )
    print("[INFO] Read branches:", read_branches, flush=True)

    with uproot.open(input_path) as f:
        df = f[tree_name].arrays(read_branches, library="pd")

    df = apply_branch_aliases(
        df=df,
        branch_map=branch_map,
        input_path=input_path,
        tree_name=tree_name,
        era=era,
    )

    print("[INFO] Loaded events:", len(df), flush=True)

    if TARGET_CATE_INDEX != -999 and CATEGORY_BRANCH in df.columns:
        before = len(df)
        df = df[df[CATEGORY_BRANCH] == TARGET_CATE_INDEX].copy()
        print(
            f"[INFO] Apply {CATEGORY_BRANCH} == {TARGET_CATE_INDEX}: {before} -> {len(df)}",
            flush=True,
        )

    if len(df) == 0:
        print("[WARNING] No events after selection. Skip writing.", flush=True)
        return None

    df = ensure_era_features(
        df,
        features,
        era,
        source_year_for_features=SOURCE_YEAR_FOR_FEATURES,
    )
    if SOURCE_YEAR_FOR_FEATURES is not None:
        print(
            f"[INFO] Model year features are filled with source_year={SOURCE_YEAR_FOR_FEATURES} "
            f"while applying to era={era}.",
            flush=True,
        )

    is_data = is_data_sample(sample_short, os.path.basename(input_path))

    df["is_data_sample"] = 1 if is_data else 0
    df["sample_hash"] = stable_hash_int((sample_short,), 999999)
    df["era_hash"] = stable_hash_int((era,), 999999)

    tree_is_syst = is_systematic_tree(tree_name, nominal_tree)
    era_is_training_era = str(era) in TRAINING_ERAS_FOR_EXACT_FOLDS
    allow_hash_for_training = (
        (not era_is_training_era)
        or (tree_is_syst and ALLOW_HASH_FOR_TRAINING_SYSTS)
        or (not STRICT_TRAINING_FOLDS)
    )
    if not era_is_training_era:
        print(
            f"[INFO] era={era} is not in TRAINING_ERAS_FOR_EXACT_FOLDS="
            f"{sorted(TRAINING_ERAS_FOR_EXACT_FOLDS)}. "
            "Training samples may use deterministic hash folds for this application year.",
            flush=True,
        )

    task_map_for_application = task_map if era_is_training_era else None
    domain_map_for_application = domain_map if era_is_training_era else None

    df = assign_application_folds(
        df=df,
        sample_short=sample_short,
        source_file=os.path.basename(input_path),
        era=era,
        is_data=is_data,
        task_map=task_map_for_application,
        domain_map=domain_map_for_application,
        allow_hash_for_training=allow_hash_for_training,
        use_exact_fold_maps=era_is_training_era,
    )

    print("[INFO] Event fold counts:", flush=True)
    print(df["event_fold_id"].value_counts().sort_index(), flush=True)

    print("[INFO] Model fold counts:", flush=True)
    print(df["model_fold_id"].value_counts().sort_index(), flush=True)

    print("[INFO] Fold method counts:", flush=True)
    print(df["fold_method"].value_counts().sort_index(), flush=True)

    df_scored = apply_models_to_df(
        df=df,
        features=features,
        models=models,
        scalers=scalers,
        tsfs=tsfs,
    )

    print("[INFO] Score summary:", flush=True)
    print(df_scored[["dnn_score", "dnn_t", "dnn_domain_score"]].describe(), flush=True)

    return make_root_output_dict(df_scored)


def apply_one_file(
    input_path,
    output_path,
    nominal_tree,
    sample_short,
    era,
    features,
    models,
    scalers,
    tsfs,
    task_map,
    domain_map,
):
    if not os.path.exists(input_path):
        print("[WARNING] Input file does not exist. Skip:", input_path, flush=True)
        return

    tree_names = discover_trees(
        input_path,
        nominal_tree=nominal_tree,
        tree_mode=ARGS.tree_mode,
    )

    print("\n" + "=" * 100, flush=True)
    print("[INFO] Input :", input_path, flush=True)
    print("[INFO] Output:", output_path, flush=True)
    print("[INFO] Nominal tree:", nominal_tree, flush=True)
    print("[INFO] Trees to process:", tree_names, flush=True)
    print("=" * 100, flush=True)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    wrote_any = False
    with uproot.recreate(output_path) as fout:
        for tree_name in tree_names:
            output_dict = score_one_tree(
                input_path=input_path,
                tree_name=tree_name,
                nominal_tree=nominal_tree,
                sample_short=sample_short,
                era=era,
                features=features,
                models=models,
                scalers=scalers,
                tsfs=tsfs,
                task_map=task_map,
                domain_map=domain_map,
            )
            if output_dict is None:
                continue
            fout[tree_name] = output_dict
            wrote_any = True
            print(f"[INFO] Wrote tree {tree_name} to {output_path}", flush=True)

    if wrote_any:
        print("[INFO] Wrote:", output_path, flush=True)
    else:
        print("[WARNING] No trees were written:", output_path, flush=True)


# ============================================================
# Main
# ============================================================

def main():
    setup_gpu()

    cfg, features = load_features(MODEL_DIR)
    print(f"[INFO] APPLY_TAG = {APPLY_TAG}", flush=True)
    models, scalers, tsfs = load_all_objects(MODEL_DIR)

    task_map, domain_map = load_training_fold_maps(MODEL_DIR)

    print("\n[INFO] Fold-map status:", flush=True)
    if task_map is None:
        print("       task map: missing", flush=True)
    else:
        print(f"       task map: has_event_id = {task_map.get('has_event_id', False)}", flush=True)

    if domain_map is None:
        print("       domain map: missing", flush=True)
    else:
        print(f"       domain map: has_event_id = {domain_map.get('has_event_id', False)}", flush=True)

    print(f"       USE_RANDOM_HASH_FOLD_FOR_APPLICATION = {USE_RANDOM_HASH_FOLD_FOR_APPLICATION}", flush=True)
    print(
        "[INFO] Training samples require exact fold-map matching; "
        "hash folds are only used for samples that were not in the relevant training loss.\n",
        flush=True,
    )

    for ipath, folder in path.items():
        era = era_names[ipath]
        nominal_tree = tree_name_by_path.get(ipath, TREE_NAME_DEFAULT)

        OUT_DIR_1 = f"simpleDNN_{APPLY_TAG}_{current_date}_{era}"
        era_out_dir = os.path.join(path[ipath], OUT_DIR_1)
        os.makedirs(era_out_dir, exist_ok=True)

        print("\n" + "#" * 100, flush=True)
        print("[INFO] Era:", era, flush=True)
        print("[INFO] Input folder:", folder, flush=True)
        print("[INFO] Output folder:", era_out_dir, flush=True)
        print("#" * 100, flush=True)

        samples = samples_by_path[ipath]

        for sample_short, filename in samples.items():
            input_path = os.path.join(folder, filename)

            # Output uses the same filename in a new directory.
            #output_path = os.path.join(era_out_dir, filename)
            #output_filename = filename.replace(".root", "_withDomainDNN.root")
            #output_filename = filename.replace(".root", "_idiso.root")
            output_filename = filename

            output_path = os.path.join(era_out_dir, output_filename)

            apply_one_file(
                input_path=input_path,
                output_path=output_path,
                nominal_tree=nominal_tree,
                sample_short=sample_short,
                era=era,
                features=features,
                models=models,
                scalers=scalers,
                tsfs=tsfs,
                task_map=task_map,
                domain_map=domain_map,
            )

    print("\n[DONE] All samples processed.", flush=True)


if __name__ == "__main__":
    main()
