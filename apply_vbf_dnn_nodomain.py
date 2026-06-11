#!/usr/bin/env python3

import os
import json
import pickle
import joblib
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import uproot
import tensorflow as tf


# ============================================================
# User configuration
# ============================================================

current_date = datetime.now().strftime("%m%d")

MODEL_DIR = "/eos/user/q/qguo/SWAN_projects/ML_test/saved_model_noDomain_4branch_run3_20260602_v1/"

N_FOLDS = 4

TREE_NAME_DEFAULT = "data_two_jet_m110To150"

TARGET_CATE_INDEX = 3
CATEGORY_BRANCH = "cate_index"

MASS_BRANCH = "diMufsr_kit_BSC_mass"
WEIGHT_BRANCH = "eventWeight"

SR_LOW = 115.0
SR_HIGH = 135.0

FIX_SB_MASS_TO_125 = True
FIXED_MASS_VALUE = 125.0

BATCH_SIZE = 16384

USE_RANDOM_HASH_FOLD_FOR_APPLICATION = True
RANDOM_FOLD_SEED_TAG = "NoDomain_4branch_application_20260611_v1"

# Your current application mapping.
# event_fold_id = 1 -> model_fold_id = 4
# event_fold_id = 2 -> model_fold_id = 3
# event_fold_id = 3 -> model_fold_id = 2
# event_fold_id = 4 -> model_fold_id = 1
#
# If future training saves fold_id directly as the test-model fold,
# change this to {1:1, 2:2, 3:3, 4:4}.
APPLICATION_MODEL_FOR_EVENT_FOLD = {
    1: 4,
    2: 3,
    3: 2,
    4: 1,
}


# ============================================================
# Samples used in no-domain SR task training
# ============================================================

TASK_TRAINING_SAMPLES = {
    "dy_inc",
    "dy_vbffilter",
    "ewk_zjj",
    "TTTo2L2Nu",
    "vbf_hmm",
    "ggh_hmm",
}


# ============================================================
# Input paths and sample list
# ============================================================

path = {}
path[0] = "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2022_ggHVBF/"
path[1] = "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2022EE_ggHVBF/"
path[2] = "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2023_ggHVBF/"
path[3] = "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2023BPix_ggHVBF/"
path[4] = "/eos/user/z/zhangxu/sharing/hmm/2024_v4/skimmed_ntuples/SRSB_v2/"
path[5] = "/eos/user/h/hakou/Hmumu_Share/qguo/2025/skimmed_ntuples/SRSB_noJetHornVeto/"

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
    4: "data_two_jet_m110To150",
    5: "data_two_jet_m110To150",
}

samples_by_path = {
    0: {
        "data": "data.root",
        "dy_inc": "DY_105To160.root",
        "dy_vbffilter": "DY_105To160_Fil-VBF_passvbffilter.root",
        "ewk_zjj": "EWK_LLJJ_M105To160.root",
        "TTTo2L2Nu": "TTTo2L2Nu.root",
        "vbf_hmm": "VBFHToMuMu_M125_ggHUnc.root",
        "ggh_hmm": "GluGluHToMuMu_M125_ggHUnc.root",
        "WWTo2L2Nu": "WWTo2L2Nu.root",
        "WWW": "WWW.root",
        "WWZ": "WWZ.root",
        "WZTo2L2Q": "WZTo2L2Q.root",
        "WZTo3LNu": "WZTo3LNu.root",
        "WZZ": "WZZ.root",
        "ZZTo2L2Nu": "ZZTo2L2Nu.root",
        "ZZTo2L2Q": "ZZTo2L2Q.root",
        "ZZTo4L": "ZZTo4L.root",
        "ZZZ": "ZZZ.root",
    },
    1: {
        "data": "data.root",
        "dy_inc": "DY_105To160.root",
        "dy_vbffilter": "DY_105To160_Fil-VBF_passvbffilter.root",
        "ewk_zjj": "EWK_LLJJ_M105To160.root",
        "TTTo2L2Nu": "TTTo2L2Nu.root",
        "vbf_hmm": "VBFHToMuMu_M125_ggHUnc.root",
        "ggh_hmm": "GluGluHToMuMu_M125_ggHUnc.root",
        "WWTo2L2Nu": "WWTo2L2Nu.root",
        "WWW": "WWW.root",
        "WWZ": "WWZ.root",
        "WZTo2L2Q": "WZTo2L2Q.root",
        "WZTo3LNu": "WZTo3LNu.root",
        "WZZ": "WZZ.root",
        "ZZTo2L2Nu": "ZZTo2L2Nu.root",
        "ZZTo2L2Q": "ZZTo2L2Q.root",
        "ZZTo4L": "ZZTo4L.root",
        "ZZZ": "ZZZ.root",
    },
    2: {
        "data": "data.root",
        "dy_inc": "DY_105To160.root",
        "dy_vbffilter": "DY_105To160_Fil-VBF_passvbffilter.root",
        "ewk_zjj": "EWK_LLJJ_M105To160.root",
        "TTTo2L2Nu": "TTTo2L2Nu.root",
        "vbf_hmm": "VBFHToMuMu_M125_ggHUnc.root",
        "ggh_hmm": "GluGluHToMuMu_M125_ggHUnc.root",
        "WWTo2L2Nu": "WWTo2L2Nu.root",
        "WWW": "WWW.root",
        "WWZ": "WWZ.root",
        "WZTo2L2Q": "WZTo2L2Q.root",
        "WZTo3LNu": "WZTo3LNu.root",
        "WZZ": "WZZ.root",
        "ZZTo2L2Nu": "ZZTo2L2Nu.root",
        "ZZTo2L2Q": "ZZTo2L2Q.root",
        "ZZTo4L": "ZZTo4L.root",
        "ZZZ": "ZZZ.root",
    },
    3: {
        "data": "data.root",
        "dy_inc": "DY_105To160.root",
        "dy_vbffilter": "DY_105To160_Fil-VBF_passvbffilter.root",
        "ewk_zjj": "EWK_LLJJ_M105To160.root",
        "TTTo2L2Nu": "TTTo2L2Nu.root",
        "vbf_hmm": "VBFHToMuMu_M125_ggHUnc.root",
        "ggh_hmm": "GluGluHToMuMu_M125_ggHUnc.root",
        "WWTo2L2Nu": "WWTo2L2Nu.root",
        "WWW": "WWW.root",
        "WWZ": "WWZ.root",
        "WZTo2L2Q": "WZTo2L2Q.root",
        "WZTo3LNu": "WZTo3LNu.root",
        "WZZ": "WZZ.root",
        "ZZTo2L2Nu": "ZZTo2L2Nu.root",
        "ZZTo2L2Q": "ZZTo2L2Q.root",
        "ZZTo4L": "ZZTo4L.root",
        "ZZZ": "ZZZ.root",
    },
    4: {
        #"data": "data.root",
        #"dy_inc": "DY_105To160_Inc_failvbffilter.root",
        #"dy_vbffilter": "DY_105To160_Fil-VBF_passvbffilter.root",
        #"ewk_zjj": "EWK_LLJJ_M105To160.root",
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
        "ST_tW_antitop": "ST_tW_antitop.root",
        "ST_tW_top": "ST_tW_top.root",
    },
    5: {
        "data": "data_25_all.root",
        "dy_inc": "DY_105To160_Inc_failvbffilter.root",
        "dy_vbffilter": "DY_105To160_Fil-VBF_passvbffilter.root",
        "ewk_zjj": "EWK_LLJJ_M105To160.root",
        "TTTo2L2Nu": "TTTo2L2Nu.root",
        "vbf_hmm": "VBFHToMuMu_M125.root",
        "ggh_hmm": "GluGluHToMuMu_M125.root",
        "WWTo2L2Nu": "WWTo2L2Nu.root",
        "WWW": "WWW.root",
        "WWZ": "WWZ.root",
        "WZTo2L2Q": "WZTo2L2Q.root",
        "WZTo3LNu": "WZTo3LNu.root",
        "WZZ": "WZZ.root",
        "ZZTo2L2Nu": "ZZTo2L2Nu.root",
        "ZZTo2L2Q": "ZZTo2L2Q.root",
        "ZZTo4L": "ZZTo4L.root",
        "ZZZ": "ZZZ.root",
    },
}

# Keep all years defined above, but run only 2024 for now.
ONLY_YEAR = "2024"
# ONLY_YEAR = None

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
    ["run", "luminosityBlock", "event"],
    ["run", "lumi", "event"],
    ["run", "luminosityBlock", "evt"],
    ["run", "lumi", "evt"],
]


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


def ensure_era_features(df, features, era):
    df = df.copy()

    for feat in features:
        if feat.startswith("era_"):
            era_value = feat.replace("era_", "")
            df[feat] = 1.0 if str(era) == era_value else 0.0

    return df


# ============================================================
# Model loading with debug
# ============================================================

def find_model_path(model_dir, model_fold):
    candidates = [
        os.path.join(model_dir, "models", f"merged_noDomain_4branch_model_fold_{model_fold}.keras"),
        os.path.join(model_dir, "models", f"merged_noDomain_4branch_model_fold_{model_fold}.h5"),
        os.path.join(model_dir, "models", f"merged_model_fold_{model_fold}.keras"),
        os.path.join(model_dir, "models", f"merged_model_fold_{model_fold}.h5"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    print("[ERROR] Tried model candidates:", flush=True)
    for p in candidates:
        print("   ", p, flush=True)

    raise RuntimeError(f"Cannot find no-domain model for model fold {model_fold} in {model_dir}")


def find_scaler_path(model_dir, model_fold):
    candidates = [
        os.path.join(model_dir, "models", f"scaler_fold_{model_fold}.pkl"),
        os.path.join(model_dir, "models", f"scaler_fold_{model_fold}.joblib"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    print("[ERROR] Tried scaler candidates:", flush=True)
    for p in candidates:
        print("   ", p, flush=True)

    raise RuntimeError(f"Cannot find scaler for model fold {model_fold} in {model_dir}")


def find_tsf_path(model_dir, model_fold):
    candidates = [
        os.path.join(model_dir, "models", f"DNN_tsf_fold_{model_fold}.pkl"),
        os.path.join(model_dir, "models", f"DNN_tsf_fold_{model_fold}.joblib"),
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


def print_model_io(model, model_fold):
    print(f"[INFO] Model fold {model_fold} input names:", flush=True)
    try:
        print("       model.input_names =", model.input_names, flush=True)
    except Exception:
        print("       model.input_names not available", flush=True)

    for i, inp in enumerate(model.inputs):
        print(f"       input[{i}] name={inp.name}, shape={inp.shape}", flush=True)

    print(f"[INFO] Model fold {model_fold} output names:", flush=True)
    try:
        print("       model.output_names =", model.output_names, flush=True)
    except Exception:
        print("       model.output_names not available", flush=True)

    for i, out in enumerate(model.outputs):
        print(f"       output[{i}] name={out.name}, shape={out.shape}", flush=True)


def load_model_with_debug(model_path, model_fold):
    print(f"[INFO] Loading model fold {model_fold}: {model_path}", flush=True)

    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False,
        )
    except TypeError as e:
        print("[WARNING] load_model(..., safe_mode=False) failed with TypeError.", flush=True)
        print("[WARNING] Try load_model(..., compile=False) without safe_mode.", flush=True)
        print("[WARNING] Error:", e, flush=True)
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
        )

    print_model_io(model, model_fold)

    return model


def load_all_objects(model_dir):
    models = {}
    scalers = {}
    tsfs = {}

    for model_fold in range(1, N_FOLDS + 1):
        model_path = find_model_path(model_dir, model_fold)
        scaler_path = find_scaler_path(model_dir, model_fold)
        tsf_path = find_tsf_path(model_dir, model_fold)

        models[model_fold] = load_model_with_debug(model_path, model_fold)

        print(f"[INFO] Loading scaler fold {model_fold}: {scaler_path}", flush=True)
        scalers[model_fold] = load_pickle_or_joblib(scaler_path)

        if tsf_path is not None:
            print(f"[INFO] Loading DNN_t transformer fold {model_fold}: {tsf_path}", flush=True)
            tsfs[model_fold] = load_pickle_or_joblib(tsf_path)
        else:
            print(f"[WARNING] No DNN_t transformer for fold {model_fold}. dnn_t will be -999.", flush=True)
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
# Optional exact task fold map matching
# ============================================================

def load_task_fold_map(model_dir):
    p = os.path.join(model_dir, "task_df_with_folds.pkl")

    if not os.path.exists(p):
        print("[WARNING] task_df_with_folds.pkl not found. Will use random/hash fallback.", flush=True)
        return None

    print(f"[INFO] Loading task fold map: {p}", flush=True)
    fold_df = pd.read_pickle(p)

    print("[INFO] task fold map loaded.", flush=True)
    print("[INFO] rows:", len(fold_df), flush=True)
    print("[INFO] columns:", list(fold_df.columns), flush=True)

    if "fold_id" not in fold_df.columns:
        print("[WARNING] task_df_with_folds.pkl has no fold_id. Will use random/hash fallback.", flush=True)
        return None

    id_cols = get_event_id_cols(fold_df)

    if id_cols is None:
        print("[WARNING] task_df_with_folds.pkl does NOT have run/lumi/event.", flush=True)
        print("[WARNING] Exact out-of-fold matching is impossible. Will use random/hash fallback.", flush=True)
        return {
            "df": None,
            "id_cols": None,
            "has_event_id": False,
        }

    keep_cols = id_cols + ["fold_id"]

    if "source_file" in fold_df.columns:
        keep_cols = ["source_file"] + keep_cols

    if "era" in fold_df.columns:
        keep_cols = ["era"] + keep_cols

    fold_df = fold_df[keep_cols].drop_duplicates()

    print("[INFO] task map supports exact event-id matching.", flush=True)
    print("[INFO] task id columns:", id_cols, flush=True)
    print("[INFO] task map size after drop_duplicates:", len(fold_df), flush=True)

    return {
        "df": fold_df,
        "id_cols": id_cols,
        "has_event_id": True,
        "has_source_file": "source_file" in fold_df.columns,
        "has_era": "era" in fold_df.columns,
    }


def exact_match_task_fold_map(df, task_map, source_file, era):
    if task_map is None:
        return df, False

    if not task_map.get("has_event_id", False):
        return df, False

    id_cols = task_map["id_cols"]

    if not all(c in df.columns for c in id_cols):
        print(f"[WARNING] ROOT input does not contain task map id columns: {id_cols}", flush=True)
        return df, False

    def _try_merge(use_source_file=True, use_era=True):
        left = df[id_cols].copy()
        left["_rowid_"] = np.arange(len(df))

        merge_cols = id_cols.copy()

        if use_source_file and task_map["has_source_file"]:
            left["source_file"] = source_file
            merge_cols = ["source_file"] + merge_cols

        if use_era and task_map["has_era"]:
            left["era"] = era
            merge_cols = ["era"] + merge_cols

        right_cols = merge_cols + ["fold_id"]

        for c in right_cols:
            if c not in task_map["df"].columns:
                return None

        right = task_map["df"][right_cols].drop_duplicates()
        merged = left.merge(right, on=merge_cols, how="left")
        merged = merged.sort_values("_rowid_")

        n_match = int(merged["fold_id"].notna().sum())

        return merged, n_match, merge_cols

    if task_map["has_source_file"]:
        trial = _try_merge(use_source_file=True, use_era=True)
        if trial is not None:
            merged, n_match, merge_cols = trial
            print(f"[INFO] task exact match using {merge_cols}: matched {n_match}/{len(df)}", flush=True)
            if n_match > 0:
                out = df.copy()
                out["event_fold_id"] = merged["fold_id"].values
                out["fold_method"] = 100
                out = add_model_fold_id(out)
                return out, True

    trial = _try_merge(use_source_file=False, use_era=True)
    if trial is not None:
        merged, n_match, merge_cols = trial
        print(f"[INFO] task exact match using {merge_cols}: matched {n_match}/{len(df)}", flush=True)
        if n_match > 0:
            out = df.copy()
            out["event_fold_id"] = merged["fold_id"].values
            out["fold_method"] = 101
            out = add_model_fold_id(out)
            return out, True

    trial = _try_merge(use_source_file=False, use_era=False)
    if trial is not None:
        merged, n_match, merge_cols = trial
        print(f"[INFO] task exact match using {merge_cols}: matched {n_match}/{len(df)}", flush=True)
        if n_match > 0:
            out = df.copy()
            out["event_fold_id"] = merged["fold_id"].values
            out["fold_method"] = 102
            out = add_model_fold_id(out)
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


def assign_application_folds(df, sample_short, source_file, era, is_data, task_map):
    df = df.copy()

    df["event_fold_id"] = np.nan
    df["model_fold_id"] = np.nan
    df["fold_method"] = -1

    use_exact_task = (
        (not is_data)
        and is_task_training_sample(sample_short)
        and task_map is not None
        and task_map.get("has_event_id", False)
    )

    if use_exact_task:
        df_mapped, used = exact_match_task_fold_map(
            df,
            task_map=task_map,
            source_file=source_file,
            era=era,
        )

        if used:
            if df_mapped["event_fold_id"].isna().any():
                n_bad = int(df_mapped["event_fold_id"].isna().sum())
                raise RuntimeError(
                    f"Exact task fold match partially failed for {sample_short}: {n_bad} missing."
                )
            return df_mapped

        raise RuntimeError(
            f"Task map has event IDs, but {sample_short} did not match. "
            "Check source_file/era/run/lumi/event."
        )

    if (
        (not is_data)
        and is_task_training_sample(sample_short)
        and task_map is not None
        and not task_map.get("has_event_id", False)
    ):
        print(
            f"[WARNING] task_df_with_folds.pkl has no event IDs. "
            f"Using random/hash fallback for task-training sample {sample_short}.",
            flush=True,
        )

    if not USE_RANDOM_HASH_FOLD_FOR_APPLICATION:
        raise RuntimeError(
            "Need random/hash fallback, but USE_RANDOM_HASH_FOLD_FOR_APPLICATION=False."
        )

    return assign_random_hash_folds(
        df,
        sample_short=sample_short,
        era=era,
        fold_method=190,
    )


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


def predict_no_domain_4branch(model, X_scaled):
    X_mass = X_scaled[:, :3].astype(np.float32)
    X_topo = X_scaled[:, 3:].astype(np.float32)

    pred = None

    # Try named inputs first.
    try:
        pred = model.predict(
            {
                "mass_res": X_mass,
                "vbf_topo": X_topo,
            },
            batch_size=BATCH_SIZE,
            verbose=0,
        )
    except Exception as e1:
        print("[WARNING] Named-input predict failed. Try list-input predict.", flush=True)
        print("[WARNING] Named-input error:", e1, flush=True)

        try:
            pred = model.predict(
                [X_mass, X_topo],
                batch_size=BATCH_SIZE,
                verbose=0,
            )
        except Exception as e2:
            print("[WARNING] Two-input predict failed. Try single full-feature input.", flush=True)
            print("[WARNING] Two-input error:", e2, flush=True)

            pred = model.predict(
                X_scaled,
                batch_size=BATCH_SIZE,
                verbose=0,
            )

    if isinstance(pred, list):
        outputs = [p.reshape(-1).astype(np.float32) for p in pred]
    else:
        arr = np.asarray(pred)

        if arr.ndim == 2 and arr.shape[1] > 1:
            outputs = [arr[:, i].reshape(-1).astype(np.float32) for i in range(arr.shape[1])]
        else:
            outputs = [arr.reshape(-1).astype(np.float32)]

    n = len(X_scaled)

    out = {
        "dnn_step1": np.full(n, -999.0, dtype=np.float32),
        "dnn_step2": np.full(n, -999.0, dtype=np.float32),
        "dnn_step3": np.full(n, -999.0, dtype=np.float32),
        "dnn_step4": np.full(n, -999.0, dtype=np.float32),
        "dnn_score": np.full(n, -999.0, dtype=np.float32),
    }

    # Expected no-domain 4-branch output:
    # [step1, step2, step3, step4, final]
    if len(outputs) >= 1:
        out["dnn_step1"] = outputs[0]
    if len(outputs) >= 2:
        out["dnn_step2"] = outputs[1]
    if len(outputs) >= 3:
        out["dnn_step3"] = outputs[2]
    if len(outputs) >= 4:
        out["dnn_step4"] = outputs[3]

    if len(outputs) >= 5:
        out["dnn_score"] = outputs[4]
    else:
        out["dnn_score"] = outputs[-1]

    return out


def apply_models_to_df(df, features, models, scalers, tsfs):
    df_out = df.copy()
    n = len(df_out)

    new_cols = [
        "dnn_step1",
        "dnn_step2",
        "dnn_step3",
        "dnn_step4",
        "dnn_score",
        "dnn_t",
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

        print(f"[INFO] Applying model fold {model_fold} to {np.sum(mask)} events", flush=True)

        df_fold = df_out.loc[mask].copy()
        X_raw, mass_info = prepare_features_for_dnn(df_fold, features)
        X_scaled = scalers[model_fold].transform(X_raw).astype(np.float32)

        pred = predict_no_domain_4branch(models[model_fold], X_scaled)

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


def build_read_branches(input_path, tree_name, features):
    available = set(get_available_branches(input_path, tree_name))

    read = set()

    for feat in features:
        if feat in available:
            read.add(feat)

    keep_extra = [
        WEIGHT_BRANCH,
        "eventWeight",
        "nominal_wgt",
        "weight",
        "signed_weight",

        MASS_BRANCH,
        "diMufsr_rc_mass",
        "diMufsr_rc_pt",
        "diMufsr_kit_BSC_pt",
        "diMufsr_kit_BSC_eta",

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
        "luminosityBlock",
        "event",
        "evt",

        "genvbffilter_flag",
        "n_jets_matched_genjet",
    ]

    for b in keep_extra:
        if b in available:
            read.add(b)

    missing_features = [
        feat for feat in features
        if feat not in available and not feat.startswith("era_")
    ]

    if missing_features:
        print("[WARNING] Missing non-era model features:", missing_features, flush=True)

    missing_extra = [
        b for b in keep_extra
        if b not in available
    ]

    if missing_extra:
        print("[INFO] Extra branches not found, skipped:", missing_extra, flush=True)

    return sorted(list(read))


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


def apply_one_file(
    input_path,
    output_path,
    tree_name,
    sample_short,
    era,
    features,
    models,
    scalers,
    tsfs,
    task_map,
):
    print("\n" + "=" * 100, flush=True)
    print("[INFO] Input :", input_path, flush=True)
    print("[INFO] Output:", output_path, flush=True)
    print("[INFO] Tree  :", tree_name, flush=True)
    print("[INFO] Sample:", sample_short, "Era:", era, flush=True)
    print("=" * 100, flush=True)

    if not os.path.exists(input_path):
        print("[WARNING] Input file does not exist. Skip:", input_path, flush=True)
        return

    read_branches = build_read_branches(input_path, tree_name, features)
    print("[INFO] Read branches:", read_branches, flush=True)

    with uproot.open(input_path) as f:
        df = f[tree_name].arrays(read_branches, library="pd")

    print("[INFO] Loaded events:", len(df), flush=True)

    if CATEGORY_BRANCH in df.columns:
        before = len(df)
        df = df[df[CATEGORY_BRANCH] == TARGET_CATE_INDEX].copy()
        print(
            f"[INFO] Apply {CATEGORY_BRANCH} == {TARGET_CATE_INDEX}: {before} -> {len(df)}",
            flush=True,
        )

    if len(df) == 0:
        print("[WARNING] No events after selection. Skip writing.", flush=True)
        return

    df = ensure_era_features(df, features, era)

    is_data = is_data_sample(sample_short, os.path.basename(input_path))

    df["is_data_sample"] = 1 if is_data else 0
    df["sample_hash"] = stable_hash_int((sample_short,), 999999)
    df["era_hash"] = stable_hash_int((era,), 999999)

    df = assign_application_folds(
        df=df,
        sample_short=sample_short,
        source_file=os.path.basename(input_path),
        era=era,
        is_data=is_data,
        task_map=task_map,
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
    print(df_scored[["dnn_score", "dnn_t"]].describe(), flush=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with uproot.recreate(output_path) as fout:
        fout[tree_name] = make_root_output_dict(df_scored)

    print("[INFO] Wrote:", output_path, flush=True)


# ============================================================
# Main
# ============================================================

def main():
    setup_gpu()

    cfg, features = load_features(MODEL_DIR)

    models, scalers, tsfs = load_all_objects(MODEL_DIR)

    task_map = load_task_fold_map(MODEL_DIR)

    print("\n[INFO] Fold-map status:", flush=True)

    if task_map is None:
        print("       task map: missing", flush=True)
    else:
        print(f"       task map: has_event_id = {task_map.get('has_event_id', False)}", flush=True)

    print(f"       USE_RANDOM_HASH_FOLD_FOR_APPLICATION = {USE_RANDOM_HASH_FOLD_FOR_APPLICATION}", flush=True)
    print("[INFO] If task map has event IDs, exact matching is used for task-training samples.", flush=True)
    print("[INFO] Otherwise random/hash fallback is used if enabled.\n", flush=True)

    for ipath, folder in path.items():
        era = era_names[ipath]
        tree_name = tree_name_by_path.get(ipath, TREE_NAME_DEFAULT)

        OUT_DIR_1 = f"dnn_noDomain_4branch_{current_date}_{era}"
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

            output_path = os.path.join(era_out_dir, filename)

            apply_one_file(
                input_path=input_path,
                output_path=output_path,
                tree_name=tree_name,
                sample_short=sample_short,
                era=era,
                features=features,
                models=models,
                scalers=scalers,
                tsfs=tsfs,
                task_map=task_map,
            )

    print("\n[DONE] All samples processed.", flush=True)


if __name__ == "__main__":
    main()
