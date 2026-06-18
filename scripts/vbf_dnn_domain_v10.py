#!/usr/bin/env python3

import os
import gc
import json
import pickle
import joblib
from datetime import datetime

import uproot
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import roc_curve, auc, roc_auc_score


# ============================================================
# Thread setup
# ============================================================

n_threads = int(os.environ.get("_CONDOR_NPROCS", os.environ.get("OMP_NUM_THREADS", "4")))
tf.config.threading.set_intra_op_parallelism_threads(n_threads)
tf.config.threading.set_inter_op_parallelism_threads(2)

print(f"TensorFlow intra_op threads = {n_threads}", flush=True)
print("TensorFlow inter_op threads = 2", flush=True)


# ============================================================
# Configuration
# ============================================================

# Save mode:
#   "both"     -> train and save both no-domain and domain models
#   "domain"   -> train and save only domain-adaptation models
#   "nodomain" -> train and save only no-domain models
SAVE_MODE = "both"

if SAVE_MODE not in ["both", "domain", "nodomain"]:
    raise RuntimeError("SAVE_MODE must be one of: both, domain, nodomain")

TRAIN_NODOMAIN = SAVE_MODE in ["both", "nodomain"]
TRAIN_DOMAIN = SAVE_MODE in ["both", "domain"]

# One path per year / era
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

signal_files_by_path = {
    0: {
        "VBFHToMuMu_M125_ggHUnc.root": "SIGNAL",
    },
    1: {
        "VBFHToMuMu_M125_ggHUnc.root": "SIGNAL",
    },
    2: {
        "VBFHToMuMu_M125_ggHUnc.root": "SIGNAL",
    },
    3: {
        "VBFHToMuMu_M125_ggHUnc.root": "SIGNAL",
    },
    4: {
        "VBFHToMuMu_M125.root": "SIGNAL",
    },
    5: {
        "VBFHToMuMu_M125.root": "SIGNAL",
    },
}

background_files_by_path = {
    0: {
        "DY_105To160.root": "DY",
        "EWK_LLJJ_M105To160.root": "EWK_ZJJ",
        "TTTo2L2Nu.root": "TT",
    },
    1: {
        "DY_105To160.root": "DY",
        "EWK_LLJJ_M105To160.root": "EWK_ZJJ",
        "TTTo2L2Nu.root": "TT",
    },
    2: {
        "DY_105To160.root": "DY",
        "EWK_LLJJ_M105To160.root": "EWK_ZJJ",
        "TTTo2L2Nu.root": "TT",
    },
    3: {
        "DY_105To160.root": "DY",
        "EWK_LLJJ_M105To160.root": "EWK_ZJJ",
        "TTTo2L2Nu.root": "TT",
    },
    4: {
        "DY_105To160_Inc_failvbffilter.root": "DY",
        "DY_105To160_Fil-VBF_passvbffilter.root": "DY",
        "EWK_LLJJ_M105To160.root": "EWK_ZJJ",
        "TTTo2L2Nu.root": "TT",
    },
    5: {
        "DY_105To160_Inc_failvbffilter.root": "DY",
        "DY_105To160_Fil-VBF_passvbffilter.root": "DY",
        "EWK_LLJJ_M105To160.root": "EWK_ZJJ",
        "TTTo2L2Nu.root": "TT",
    },
}

data_files_by_path = {
    0: {
        "data.root": "DATA",
    },
    1: {
        "data.root": "DATA",
    },
    2: {
        "data.root": "DATA",
    },
    3: {
        "data.root": "DATA",
    },
    4: {
        "data.root": "DATA",
    },
    5: {
        "data_25_all.root": "DATA",
    },
}

# Quick switch: train one year only
ONLY_YEAR = "2024"
# ONLY_YEAR = None

if ONLY_YEAR is not None:
    keep_keys = [k for k, v in era_names.items() if v == ONLY_YEAR]

    path = {k: v for k, v in path.items() if k in keep_keys}
    era_names = {k: v for k, v in era_names.items() if k in keep_keys}
    tree_name_by_path = {k: v for k, v in tree_name_by_path.items() if k in keep_keys}
    signal_files_by_path = {k: v for k, v in signal_files_by_path.items() if k in keep_keys}
    background_files_by_path = {k: v for k, v in background_files_by_path.items() if k in keep_keys}
    data_files_by_path = {k: v for k, v in data_files_by_path.items() if k in keep_keys}


# ============================================================
# Branches
# ============================================================

target_cate_index = 3

mass_branch = "diMufsr_kit_BSC_mass"
weight_branch = "eventWeight"

sr_low = 115.0
sr_high = 135.0

FIX_DOMAIN_MASS_TO_125 = True
DOMAIN_FIXED_MASS_VALUE = 125.0

# Event IDs to save for exact application later
EVENT_ID_CANDIDATES = [
    #["run", "luminosityBlock", "event"],
    ["run", "lumi", "event"],
    #["run", "luminosityBlock", "evt"],
    #["run", "lumi", "evt"],
]

extra_save_branches = [
    "run",
    "lumi",
    "event",
    "source_year",
    "genvbffilter_flag",
    "n_jets_matched_genjet",
]

branches = [
    weight_branch,

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

    "cate_index",
] + extra_save_branches

# First 3 features are mass/resolution inputs.
# Remaining features are topology inputs.
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


# ============================================================
# Training hyperparameters
# ============================================================

n_folds = 4
epochs = 40
batch_size = 4096
learning_rate = 1e-3
dropout = 0.2

# Domain adaptation parameters
alpha = 2.0
max_lambda = 0.7
focus_weight_max = 3.0
warmup_epochs = 5

# Loss weights for auxiliary tasks
loss_weight_step1 = 0.25
loss_weight_step2 = 0.25
loss_weight_step3 = 0.50
loss_weight_step4 = 0.25
loss_weight_final = 1.00


# ============================================================
# Output
# ============================================================

current_date = datetime.now().strftime("%m%d")

output_name = f"saved_model_branch4_DAandNoDomain_run3_2026{current_date}_signedPhysW_{SAVE_MODE}"
path_out = "/eos/user/q/qguo/SWAN_projects/ML_test/"
output_dir = os.path.join(path_out, output_name)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/models", exist_ok=True)
os.makedirs(f"{output_dir}/plots", exist_ok=True)
os.makedirs(f"{output_dir}/arrays", exist_ok=True)
os.makedirs(f"{output_dir}/comparison_plots", exist_ok=True)

print("output_dir =", output_dir, flush=True)
print("SAVE_MODE =", SAVE_MODE, flush=True)
print("TRAIN_NODOMAIN =", TRAIN_NODOMAIN, flush=True)
print("TRAIN_DOMAIN =", TRAIN_DOMAIN, flush=True)


# ============================================================
# Helpers
# ============================================================

def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def get_event_id_cols(df_in):
    for cols in EVENT_ID_CANDIDATES:
        if all(c in df_in.columns for c in cols):
            return cols
    return None


# ============================================================
# ROOT loading
# ============================================================

def build_file_records():
    records = []

    for ipath, folder in path.items():
        era = era_names[ipath]

        if ipath not in tree_name_by_path:
            raise RuntimeError(f"No tree name configured for path[{ipath}]")

        tree_name_this = tree_name_by_path[ipath]

        for fname, process in signal_files_by_path.get(ipath, {}).items():
            records.append({
                "path": os.path.join(folder, fname),
                "era": era,
                "ipath": ipath,
                "tree_name": tree_name_this,
                "sample_type": "signal",
                "process": process,
            })

        for fname, process in background_files_by_path.get(ipath, {}).items():
            records.append({
                "path": os.path.join(folder, fname),
                "era": era,
                "ipath": ipath,
                "tree_name": tree_name_this,
                "sample_type": "background",
                "process": process,
            })

        for fname, process in data_files_by_path.get(ipath, {}).items():
            records.append({
                "path": os.path.join(folder, fname),
                "era": era,
                "ipath": ipath,
                "tree_name": tree_name_this,
                "sample_type": "data",
                "process": process,
            })

    if len(records) == 0:
        raise RuntimeError("No file records were built. Check path and sample dictionaries.")

    return records


def load_data_from_records(records, branches):
    dfs = []

    for rec in records:
        fpath = rec["path"]
        tree_name_this = rec["tree_name"]

        if not os.path.exists(fpath):
            print("[WARNING] missing file:", fpath, flush=True)
            continue

        print("[INFO] loading:", fpath, flush=True)
        print("       tree:", tree_name_this, flush=True)

        with uproot.open(fpath) as f:
            if tree_name_this not in f:
                print("[WARNING] tree not found:", tree_name_this, "in", fpath, flush=True)
                print("[INFO] available keys:", list(f.keys()), flush=True)
                continue

            tree = f[tree_name_this]
            available = set(tree.keys())

            read_branches = [b for b in branches if b in available]

            missing = [b for b in branches if b not in available]
            if len(missing) > 0:
                print("[WARNING] missing branches in", fpath, flush=True)
                print(missing, flush=True)

            df_tmp = tree.arrays(read_branches, library="pd")

        df_tmp["era"] = rec["era"]
        df_tmp["ipath"] = rec["ipath"]
        df_tmp["tree_name"] = rec["tree_name"]
        df_tmp["sample_type"] = rec["sample_type"]
        df_tmp["process"] = rec["process"]
        df_tmp["source_file"] = os.path.basename(fpath)

        dfs.append(df_tmp)

    if len(dfs) == 0:
        raise RuntimeError("No files loaded. Please check path[], tree names, and filenames.")

    return pd.concat(dfs, ignore_index=True)


records = build_file_records()
df = load_data_from_records(records, branches)

print("Loaded events:", len(df), flush=True)


# ============================================================
# Basic selections and cleaning
# ============================================================

if "cate_index" in df.columns:
    before = len(df)
    df = df[df["cate_index"] == target_cate_index].copy()
    print(f"After cate_index == {target_cate_index}: {before} -> {len(df)}", flush=True)

if weight_branch not in df.columns:
    raise RuntimeError(f"Missing weight branch: {weight_branch}")

# Physical signed weight
df["signed_weight"] = df[weight_branch].astype(float)
df.loc[df["sample_type"] == "data", "signed_weight"] = 1.0

df["signed_weight"] = df["signed_weight"].replace([np.inf, -np.inf], np.nan)
df["signed_weight"] = df["signed_weight"].fillna(0.0)

# For plotting/evaluation yields, keep signed physics weight
df["weight"] = df["signed_weight"]

df["is_sr"] = ((df[mass_branch] > sr_low) & (df[mass_branch] < sr_high)).astype(int)
df["is_sb"] = ((df[mass_branch] < sr_low) | (df[mass_branch] > sr_high)).astype(int)

# Year one-hot tags
for era in era_names.values():
    df[f"era_{era}"] = (df["era"] == era).astype(float)

year_features = [f"era_{era}" for era in era_names.values()]
all_feature_branches = feature_branches + year_features

for col in all_feature_branches:
    if col not in df.columns:
        raise RuntimeError(f"Missing feature branch: {col}")

df[all_feature_branches] = df[all_feature_branches].replace([np.inf, -np.inf], np.nan)
df[all_feature_branches] = df[all_feature_branches].fillna(-1.0)

mass_input_indices = [0, 1, 2]
topo_input_indices = list(range(3, len(all_feature_branches)))

with open(f"{output_dir}/features.json", "w") as f:
    json.dump(to_jsonable({
        "feature_branches": feature_branches,
        "year_features": year_features,
        "all_feature_branches": all_feature_branches,
        "mass_input_indices": mass_input_indices,
        "topo_input_indices": topo_input_indices,
        "mass_branch": mass_branch,
        "weight_branch": weight_branch,
        "sr_low": sr_low,
        "sr_high": sr_high,
        "target_cate_index": target_cate_index,
        "era_names": era_names,
        "tree_name_by_path": tree_name_by_path,
        "signal_files_by_path": signal_files_by_path,
        "background_files_by_path": background_files_by_path,
        "data_files_by_path": data_files_by_path,
        "SAVE_MODE": SAVE_MODE,
        "TRAIN_NODOMAIN": TRAIN_NODOMAIN,
        "TRAIN_DOMAIN": TRAIN_DOMAIN,
        "FIX_DOMAIN_MASS_TO_125": FIX_DOMAIN_MASS_TO_125,
        "DOMAIN_FIXED_MASS_VALUE": DOMAIN_FIXED_MASS_VALUE,
        "weight_treatment": "signed physical weights; BCE normalized by sum(abs(weight))",
        "branch_inputs": {
            "step1": "mass_res + vbf_topo",
            "step2": "mass_res + vbf_topo",
            "step3": "vbf_topo only",
            "step4": "mass_res only",
            "final": "merged latent features",
            "domain": "shared latent features through GRL",
        },
    }), f, indent=2)


# ============================================================
# Dataset construction
# ============================================================

# Step 1: signal vs EWK-ZJJ in SR
step1_df = df[
    (df["is_sr"] == 1)
    & (
        ((df["sample_type"] == "signal") & (df["process"] == "SIGNAL"))
        | ((df["sample_type"] == "background") & (df["process"] == "EWK_ZJJ"))
    )
].copy()
step1_df["label_step1"] = (step1_df["sample_type"] == "signal").astype(float)

# Step 2: signal vs DY in SR
step2_df = df[
    (df["is_sr"] == 1)
    & (
        ((df["sample_type"] == "signal") & (df["process"] == "SIGNAL"))
        | ((df["sample_type"] == "background") & (df["process"] == "DY"))
    )
].copy()
step2_df["label_step2"] = (step2_df["sample_type"] == "signal").astype(float)

# Step 3 / Step 4 / Final task: signal vs all background in SR
task_df = df[
    (df["is_sr"] == 1)
    & (df["sample_type"].isin(["signal", "background"]))
].copy()
task_df["task_label"] = (task_df["sample_type"] == "signal").astype(float)

# Domain: data SB vs background MC SB
domain_df = df[
    (df["is_sb"] == 1)
    & (df["sample_type"].isin(["data", "background"]))
].copy()
domain_df["domain_label"] = (domain_df["sample_type"] == "data").astype(float)

if len(step1_df) == 0:
    raise RuntimeError("step1_df is empty. Check EWK_ZJJ file/process labels and SR selection.")
if len(step2_df) == 0:
    raise RuntimeError("step2_df is empty. Check DY file/process labels and SR selection.")
if len(task_df) == 0:
    raise RuntimeError("task_df is empty. Check signal/background and SR selection.")
if TRAIN_DOMAIN and len(domain_df) == 0:
    raise RuntimeError("domain_df is empty. Add data files and check SB selection.")

print("\n[Dataset sizes]", flush=True)

print("Step1 signal vs EWK_ZJJ:", len(step1_df), flush=True)
print(step1_df.groupby(["sample_type", "process"])["signed_weight"].sum(), flush=True)

print("\nStep2 signal vs DY:", len(step2_df), flush=True)
print(step2_df.groupby(["sample_type", "process"])["signed_weight"].sum(), flush=True)

print("\nTask signal vs all background:", len(task_df), flush=True)
print(task_df.groupby(["sample_type", "process"])["signed_weight"].sum(), flush=True)

print("\nDomain data vs MC in SB:", len(domain_df), flush=True)
print(domain_df.groupby(["sample_type", "process"])["signed_weight"].sum(), flush=True)


# ============================================================
# Training weights: signed physical weights
# ============================================================

def add_signed_train_weight(df_in, label_col):
    """
    Use signed physical weights in BCE.

    Important:
      - Negative MC weights are kept negative.
      - Per-class normalization uses sum(abs(weight)) to keep scale stable.
      - BCE denominator uses sum(abs(weight)).
    """
    out = df_in.copy()
    out["train_weight"] = out["signed_weight"].astype(float)

    labels = sorted(out[label_col].unique())
    n_class = len(labels)

    for label in labels:
        mask = out[label_col] == label
        sum_abs_w = np.sum(np.abs(out.loc[mask, "train_weight"].values))

        if sum_abs_w > 0:
            out.loc[mask, "train_weight"] *= len(out) / (n_class * sum_abs_w)

    out["train_weight"] = out["train_weight"].replace([np.inf, -np.inf], np.nan)
    out["train_weight"] = out["train_weight"].fillna(0.0)

    return out


step1_df = add_signed_train_weight(step1_df, "label_step1")
step2_df = add_signed_train_weight(step2_df, "label_step2")
task_df = add_signed_train_weight(task_df, "task_label")
domain_df = add_signed_train_weight(domain_df, "domain_label")


# ============================================================
# Folds
# ============================================================

def assign_folds(df_in, n_folds=4, random_state=42):
    out = df_in.copy()
    out["fold_id"] = -1

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    indices = np.arange(len(out))

    for fold, (_, test_idx) in enumerate(kf.split(indices), start=1):
        out.iloc[test_idx, out.columns.get_loc("fold_id")] = fold

    return out


def split_by_fold(df_in, fold, label_col):
    train_val = df_in[df_in["fold_id"] != fold].copy()
    test = df_in[df_in["fold_id"] == fold].copy()

    train, val = train_test_split(
        train_val,
        test_size=0.25,
        random_state=100 + fold,
        stratify=train_val[label_col],
    )

    return train.copy(), val.copy(), test.copy()


step1_df = assign_folds(step1_df, n_folds=n_folds, random_state=11)
step2_df = assign_folds(step2_df, n_folds=n_folds, random_state=22)
task_df = assign_folds(task_df, n_folds=n_folds, random_state=33)
domain_df = assign_folds(domain_df, n_folds=n_folds, random_state=44)

# Save with run/lumi/event if available
step1_df.to_pickle(f"{output_dir}/step1_df_with_folds.pkl")
step2_df.to_pickle(f"{output_dir}/step2_df_with_folds.pkl")
task_df.to_pickle(f"{output_dir}/task_df_with_folds.pkl")
domain_df.to_pickle(f"{output_dir}/domain_df_with_folds.pkl")

event_id_cols_task = get_event_id_cols(task_df)
event_id_cols_domain = get_event_id_cols(domain_df)

print("[INFO] task event id columns:", event_id_cols_task, flush=True)
print("[INFO] domain event id columns:", event_id_cols_domain, flush=True)

if event_id_cols_task is None:
    print("[WARNING] No run/lumi/event columns found in task_df. Exact application matching will not be possible.", flush=True)
if event_id_cols_domain is None:
    print("[WARNING] No run/lumi/event columns found in domain_df. Exact application matching will not be possible.", flush=True)


# ============================================================
# Gradient reversal
# ============================================================

@tf.custom_gradient
def gradient_reverse(x, lambd):
    def grad(dy):
        return -lambd * dy, None
    return tf.identity(x), grad


class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lambd = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def call(self, x, training=None):
        return gradient_reverse(x, self.lambd)


def lambda_schedule(epoch, max_epochs, max_lambda=1.0, warmup_epochs=5):
    if epoch <= warmup_epochs:
        return 0.0

    p = (epoch - warmup_epochs) / float(max_epochs - warmup_epochs)
    p = np.clip(p, 0.0, 1.0)

    return max_lambda * (2.0 / (1.0 + np.exp(-6.0 * p)) - 1.0)


# ============================================================
# Model
# ============================================================

def dense_block(x, name, dropout=0.2):
    x = tf.keras.layers.Dense(64, activation="relu", name=f"{name}_dense1")(x)
    x = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout1")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name=f"{name}_dense2")(x)
    x = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout2")(x)
    x = tf.keras.layers.Dense(16, activation="relu", name=f"{name}_latent")(x)
    x = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout3")(x)
    return x


def build_branch4_model(n_mass, n_topo, dropout=0.2, use_domain=True):
    input_mass_res = tf.keras.layers.Input(shape=(n_mass,), name="mass_res")
    input_vbf_topo = tf.keras.layers.Input(shape=(n_topo,), name="vbf_topo")

    # Branch 1: Signal vs EWK-ZJJ, mass + topo
    x1_in = tf.keras.layers.Concatenate(name="step1_input")([input_mass_res, input_vbf_topo])
    x1 = dense_block(x1_in, name="step1_signal_vs_ewkzjj", dropout=dropout)
    step1_output = tf.keras.layers.Dense(1, activation="sigmoid", name="Step1_EWKZJJ_Output")(x1)

    # Branch 2: Signal vs DY, mass + topo
    x2_in = tf.keras.layers.Concatenate(name="step2_input")([input_mass_res, input_vbf_topo])
    x2 = dense_block(x2_in, name="step2_signal_vs_dy", dropout=dropout)
    step2_output = tf.keras.layers.Dense(1, activation="sigmoid", name="Step2_DY_Output")(x2)

    # Branch 3: Signal vs all background, vbf topo only
    x3 = dense_block(input_vbf_topo, name="step3_no_mass_all_bkg", dropout=dropout)
    step3_output = tf.keras.layers.Dense(1, activation="sigmoid", name="Step3_NoMass_Output")(x3)

    # Branch 4: Signal vs all background, mass only
    x4 = dense_block(input_mass_res, name="step4_mass_only_all_bkg", dropout=dropout)
    step4_output = tf.keras.layers.Dense(1, activation="sigmoid", name="Step4_MassOnly_Output")(x4)

    # Final merged task head
    merged_features = tf.keras.layers.Concatenate(name="merged_features")([x1, x2, x3, x4])

    shared = tf.keras.layers.Dense(64, activation="relu", name="shared_dense1")(merged_features)
    shared = tf.keras.layers.Dropout(dropout, name="shared_dropout1")(shared)
    shared = tf.keras.layers.Dense(32, activation="relu", name="shared_dense2")(shared)
    shared = tf.keras.layers.Dropout(dropout, name="shared_dropout2")(shared)
    shared = tf.keras.layers.Dense(16, activation="relu", name="shared_latent")(shared)

    final_task = tf.keras.layers.Dense(16, activation="relu", name="final_task_dense")(shared)
    final_task_output = tf.keras.layers.Dense(1, activation="sigmoid", name="Final_Task_Output")(final_task)

    outputs = [
        step1_output,
        step2_output,
        step3_output,
        step4_output,
        final_task_output,
    ]

    grl = None

    if use_domain:
        grl = GradientReversalLayer(name="Gradient_Reversal")
        reversed_shared = grl(shared)

        domain = tf.keras.layers.Dense(16, activation="relu", name="domain_dense1")(reversed_shared)
        domain = tf.keras.layers.Dropout(dropout, name="domain_dropout1")(domain)
        domain_output = tf.keras.layers.Dense(1, activation="sigmoid", name="Domain_Output")(domain)

        outputs.append(domain_output)

        model_name = "VBF_DNN_with_Domain_Adaptation"
    else:
        model_name = "VBF_DNN_noDomain_4branch"

    model = tf.keras.Model(
        inputs=[input_mass_res, input_vbf_topo],
        outputs=outputs,
        name=model_name,
    )

    return model, grl


# ============================================================
# Array helpers
# ============================================================

def fit_scaler_from_task_train(task_train):
    scaler = StandardScaler()
    X = task_train[all_feature_branches].values
    scaler.fit(X)
    return scaler


def copy_fix_domain_mass(df_in):
    out = df_in.copy()

    if FIX_DOMAIN_MASS_TO_125:
        out[mass_branch] = DOMAIN_FIXED_MASS_VALUE

    return out


def transform_df(df_in, scaler, fix_domain_mass=False):
    if fix_domain_mass:
        df_work = copy_fix_domain_mass(df_in)
    else:
        df_work = df_in

    X = df_work[all_feature_branches].values
    X_scaled = scaler.transform(X)

    X_mass = X_scaled[:, :3]
    X_topo = X_scaled[:, 3:]

    return X_mass.astype(np.float32), X_topo.astype(np.float32)


def make_batches(X_mass, X_topo, y, w, batch_size, shuffle=True):
    n = len(y)
    idx = np.arange(n)

    if shuffle:
        np.random.shuffle(idx)

    for start in range(0, n, batch_size):
        sub = idx[start:start + batch_size]

        yield (
            X_mass[sub].astype(np.float32),
            X_topo[sub].astype(np.float32),
            y[sub].astype(np.float32).reshape(-1, 1),
            w[sub].astype(np.float32).reshape(-1, 1),
        )


def bce_loss_signed_weight(y_true, y_pred, weight):
    """
    Signed physical weights are allowed.

    The numerator uses signed weights:
        sum(BCE * signed_weight)

    The normalization uses sum(abs(weight)):
        sum(abs(signed_weight))

    This keeps the signed-weight gradient but avoids zero/negative denominator.
    """
    loss = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    loss = tf.reshape(loss, (-1, 1))

    numerator = tf.reduce_sum(loss * weight)
    denominator = tf.reduce_sum(tf.abs(weight)) + 1e-8

    return numerator / denominator


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    grl,
    optimizer,
    arrays,
    epoch,
    max_epochs,
    use_domain=True,
    batch_size=2048,
    alpha=5.0,
    max_lambda=1.0,
):
    if use_domain:
        grl_lambda = lambda_schedule(
            epoch,
            max_epochs,
            max_lambda=max_lambda,
            warmup_epochs=warmup_epochs,
        )
        grl.lambd.assign(grl_lambda)
    else:
        grl_lambda = 0.0

    n_step1 = int(np.ceil(len(arrays["step1"][2]) / batch_size))
    n_step2 = int(np.ceil(len(arrays["step2"][2]) / batch_size))
    n_task = int(np.ceil(len(arrays["task"][2]) / batch_size))

    if use_domain:
        n_domain = int(np.ceil(len(arrays["domain"][2]) / batch_size))
        n_steps = min(n_step1, n_step2, n_task, n_domain)
    else:
        n_steps = min(n_step1, n_step2, n_task)

    if n_steps <= 0:
        raise RuntimeError("n_steps <= 0. One of the training datasets is empty.")

    step1_iter = make_batches(*arrays["step1"], batch_size=batch_size, shuffle=True)
    step2_iter = make_batches(*arrays["step2"], batch_size=batch_size, shuffle=True)
    task_iter = make_batches(*arrays["task"], batch_size=batch_size, shuffle=True)

    if use_domain:
        domain_iter = make_batches(*arrays["domain"], batch_size=batch_size, shuffle=True)
    else:
        domain_iter = None

    losses = {
        "total": [],
        "step1": [],
        "step2": [],
        "step3": [],
        "step4": [],
        "final": [],
        "domain": [],
    }

    for i in range(n_steps):
        s1_m, s1_t, s1_y, s1_w = next(step1_iter)
        s2_m, s2_t, s2_y, s2_w = next(step2_iter)
        tk_m, tk_t, tk_y, tk_w = next(task_iter)

        if use_domain:
            dm_m, dm_t, dm_y, dm_w = next(domain_iter)

        with tf.GradientTape() as tape:
            s1_outputs = model({"mass_res": s1_m, "vbf_topo": s1_t}, training=True)
            s2_outputs = model({"mass_res": s2_m, "vbf_topo": s2_t}, training=True)
            tk_outputs = model({"mass_res": tk_m, "vbf_topo": tk_t}, training=True)

            s1_pred = s1_outputs[0]
            s2_pred = s2_outputs[1]
            step3_pred = tk_outputs[2]
            step4_pred = tk_outputs[3]
            final_pred = tk_outputs[4]

            loss_step1 = bce_loss_signed_weight(s1_y, s1_pred, s1_w)
            loss_step2 = bce_loss_signed_weight(s2_y, s2_pred, s2_w)
            loss_step3 = bce_loss_signed_weight(tk_y, step3_pred, tk_w)
            loss_step4 = bce_loss_signed_weight(tk_y, step4_pred, tk_w)
            loss_final = bce_loss_signed_weight(tk_y, final_pred, tk_w)

            loss_task_all = (
                loss_weight_step1 * loss_step1
                + loss_weight_step2 * loss_step2
                + loss_weight_step3 * loss_step3
                + loss_weight_step4 * loss_step4
                + loss_weight_final * loss_final
            )

            if use_domain:
                dm_outputs = model({"mass_res": dm_m, "vbf_topo": dm_t}, training=True)
                dm_task_score = dm_outputs[4]
                dm_domain_pred = dm_outputs[5]

                focus_weight = 1.0 + alpha * tf.square(tf.stop_gradient(dm_task_score))
                focus_weight = tf.clip_by_value(focus_weight, 1.0, focus_weight_max)

                # signed physical weight * focus
                domain_weight = dm_w * focus_weight

                loss_domain = bce_loss_signed_weight(dm_y, dm_domain_pred, domain_weight)

                domain_contribution = grl_lambda * tf.math.log1p(loss_domain)
                loss_total = loss_task_all + domain_contribution
            else:
                loss_domain = tf.constant(0.0, dtype=tf.float32)
                loss_total = loss_task_all

        grads = tape.gradient(loss_total, model.trainable_variables)
        grads_and_vars = [
            (g, v) for g, v in zip(grads, model.trainable_variables)
            if g is not None
        ]
        optimizer.apply_gradients(grads_and_vars)

        losses["total"].append(float(loss_total.numpy()))
        losses["step1"].append(float(loss_step1.numpy()))
        losses["step2"].append(float(loss_step2.numpy()))
        losses["step3"].append(float(loss_step3.numpy()))
        losses["step4"].append(float(loss_step4.numpy()))
        losses["final"].append(float(loss_final.numpy()))
        losses["domain"].append(float(loss_domain.numpy()))

        if (i + 1) % 100 == 0:
            print(
                f"    epoch {epoch:03d}: finished step {i + 1}/{n_steps}",
                flush=True,
            )

    out = {k: float(np.mean(v)) for k, v in losses.items()}
    out["lambda"] = float(grl_lambda)
    out["n_steps"] = int(n_steps)

    return out


# ============================================================
# Prediction / evaluation / plots
# ============================================================

def predict_outputs(model, X_mass, X_topo, use_domain=True, batch_size=4096):
    n_out = 6 if use_domain else 5
    outs = [[] for _ in range(n_out)]

    for start in range(0, len(X_mass), batch_size):
        xm = X_mass[start:start + batch_size].astype(np.float32)
        xt = X_topo[start:start + batch_size].astype(np.float32)

        pred = model.predict(
            {"mass_res": xm, "vbf_topo": xt},
            verbose=0,
            batch_size=batch_size,
        )

        if not isinstance(pred, list):
            pred = [pred]

        for i in range(n_out):
            outs[i].append(pred[i].reshape(-1))

    return [np.concatenate(x) for x in outs]


def safe_auc(y, score, w):
    try:
        return roc_auc_score(y, score, sample_weight=np.abs(w))
    except Exception:
        return np.nan


def plot_roc(y, score, w, title, outpath):
    try:
        fpr, tpr, _ = roc_curve(y, score, sample_weight=np.abs(w))
        roc_auc = auc(fpr, tpr)
    except Exception:
        return

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.5f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_score_distribution(y, score, w, title, outpath, names=("background", "signal")):
    plt.figure(figsize=(8, 6))

    for label, name in [(0, names[0]), (1, names[1])]:
        mask = y == label
        if np.sum(mask) == 0:
            continue

        plt.hist(
            score[mask],
            bins=50,
            weights=w[mask],
            density=True,
            histtype="step",
            linewidth=1.5,
            label=name,
        )

    plt.xlabel("DNN score")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_training_history(history, fold, tag):
    epochs_arr = np.arange(1, len(history["loss_total"]) + 1)

    plt.figure(figsize=(9, 6))
    plt.plot(epochs_arr, history["loss_total"], label="total")
    plt.plot(epochs_arr, history["loss_step1"], label="step1: S vs EWK-ZJJ")
    plt.plot(epochs_arr, history["loss_step2"], label="step2: S vs DY")
    plt.plot(epochs_arr, history["loss_step3"], label="step3: no mass")
    plt.plot(epochs_arr, history["loss_step4"], label="step4: mass only")
    plt.plot(epochs_arr, history["loss_final"], label="final")
    if "loss_domain" in history:
        plt.plot(epochs_arr, history["loss_domain"], label="domain")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training losses, {tag}, fold {fold}")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/losses_{tag}_fold_{fold}.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_arr, history["auc_final_val"], label="final task val AUC")
    if "auc_domain_val" in history:
        plt.plot(epochs_arr, history["auc_domain_val"], label="domain val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(f"AUC history, {tag}, fold {fold}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/auc_history_{tag}_fold_{fold}.png")
    plt.close()

    if "lambda" in history:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs_arr, history["lambda"], label="domain lambda")
        plt.xlabel("Epoch")
        plt.ylabel("lambda")
        plt.title(f"GRL lambda schedule, {tag}, fold {fold}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/lambda_{tag}_fold_{fold}.png")
        plt.close()


def plot_sb_data_mc_ratio(domain_eval_df, task_score, outpath, n_bins=10):
    tmp = domain_eval_df.copy()
    tmp["task_score"] = task_score

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    data_y = []
    mc_y = []
    ratio = []
    ratio_err = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (tmp["task_score"] >= lo) & (tmp["task_score"] < hi)

        data_mask = in_bin & (tmp["sample_type"] == "data")
        mc_mask = in_bin & (tmp["sample_type"] == "background")

        n_data = tmp.loc[data_mask, "signed_weight"].sum()
        n_mc = tmp.loc[mc_mask, "signed_weight"].sum()

        data_y.append(n_data)
        mc_y.append(n_mc)

        if n_mc > 0:
            r = n_data / n_mc
            err = abs(r) * np.sqrt(1.0 / max(abs(n_data), 1.0) + 1.0 / max(abs(n_mc), 1.0))
        else:
            r = np.nan
            err = np.nan

        ratio.append(r)
        ratio_err.append(err)

    data_y = np.array(data_y)
    mc_y = np.array(mc_y)
    ratio = np.array(ratio)
    ratio_err = np.array(ratio_err)

    mc_y_scaled = mc_y.copy()
    if mc_y.sum() > 0:
        mc_y_scaled = mc_y * data_y.sum() / mc_y.sum()

    plt.figure(figsize=(8, 7))

    ax1 = plt.subplot(2, 1, 1)
    ax1.step(centers, data_y, where="mid", label="Data SB")
    ax1.step(centers, mc_y_scaled, where="mid", label="MC SB, norm to data")
    ax1.set_ylabel("Events")
    ax1.set_title("SB Data/MC vs final task DNN score")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    good = np.isfinite(ratio) & np.isfinite(ratio_err) & (ratio_err >= 0)
    ax2.errorbar(centers[good], ratio[good], yerr=ratio_err[good], fmt="o")
    ax2.axhline(1.0, linestyle="--", color="black")
    ax2.set_xlabel("Final task DNN score")
    ax2.set_ylabel("Data / MC")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_score_vs_mass(df_eval, score, title, outpath):
    plt.figure(figsize=(8, 6))
    plt.hist2d(df_eval[mass_branch].values, score, bins=(60, 60))
    plt.xlabel(r"$m_{\mu\mu}$")
    plt.ylabel("DNN score")
    plt.title(title)
    plt.colorbar(label="Events")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ============================================================
# Train one model variant
# ============================================================

def train_variant_for_fold(
    fold,
    tag,
    use_domain,
    scaler,
    arrays_train,
    arrays_val,
    arrays_test,
    dfs_val_test,
):
    print("\n" + "-" * 90, flush=True)
    print(f"Start variant: {tag}, fold {fold}, use_domain={use_domain}", flush=True)
    print("-" * 90, flush=True)

    model, grl = build_branch4_model(
        n_mass=3,
        n_topo=len(all_feature_branches) - 3,
        dropout=dropout,
        use_domain=use_domain,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.summary(print_fn=lambda x: print(x, flush=True))

    history = {
        "loss_total": [],
        "loss_step1": [],
        "loss_step2": [],
        "loss_step3": [],
        "loss_step4": [],
        "loss_final": [],
        "loss_domain": [],
        "lambda": [],
        "auc_final_val": [],
    }

    if use_domain:
        history["auc_domain_val"] = []

    best_auc = -999.0

    if use_domain:
        best_model_path = f"{output_dir}/models/merged_DA_model_fold_{fold}.keras"
    else:
        best_model_path = f"{output_dir}/models/merged_noDomain_4branch_model_fold_{fold}.keras"

    for epoch in range(1, epochs + 1):
        print(f"{tag} fold {fold} epoch {epoch}: start train_one_epoch", flush=True)

        metrics = train_one_epoch(
            model=model,
            grl=grl,
            optimizer=optimizer,
            arrays=arrays_train,
            epoch=epoch,
            max_epochs=epochs,
            use_domain=use_domain,
            batch_size=batch_size,
            alpha=alpha,
            max_lambda=max_lambda,
        )

        print(f"{tag} fold {fold} epoch {epoch}: finished train_one_epoch", flush=True)

        tk_val_m, tk_val_t = arrays_val["task_X"]
        task_val = dfs_val_test["task_val"]

        val_outputs = predict_outputs(model, tk_val_m, tk_val_t, use_domain=use_domain)
        val_final_score = val_outputs[4]

        auc_final_val = safe_auc(
            task_val["task_label"].values,
            val_final_score,
            task_val["train_weight"].values,
        )

        if use_domain:
            dm_val_m, dm_val_t = arrays_val["domain_X"]
            domain_val = dfs_val_test["domain_val"]

            val_domain_outputs = predict_outputs(model, dm_val_m, dm_val_t, use_domain=True)
            val_domain_score = val_domain_outputs[5]

            auc_domain_val = safe_auc(
                domain_val["domain_label"].values,
                val_domain_score,
                domain_val["train_weight"].values,
            )
        else:
            auc_domain_val = np.nan

        history["loss_total"].append(metrics["total"])
        history["loss_step1"].append(metrics["step1"])
        history["loss_step2"].append(metrics["step2"])
        history["loss_step3"].append(metrics["step3"])
        history["loss_step4"].append(metrics["step4"])
        history["loss_final"].append(metrics["final"])
        history["loss_domain"].append(metrics["domain"])
        history["lambda"].append(metrics["lambda"])
        history["auc_final_val"].append(auc_final_val)

        if use_domain:
            history["auc_domain_val"].append(auc_domain_val)

        print(
            f"{tag} Fold {fold} | Epoch {epoch:03d} | "
            f"steps={metrics['n_steps']} | "
            f"loss={metrics['total']:.5f} | "
            f"s1={metrics['step1']:.5f} | "
            f"s2={metrics['step2']:.5f} | "
            f"s3={metrics['step3']:.5f} | "
            f"s4={metrics['step4']:.5f} | "
            f"final={metrics['final']:.5f} | "
            f"domain={metrics['domain']:.5f} | "
            f"lambda={metrics['lambda']:.3f} | "
            f"val_final_auc={auc_final_val:.5f} | "
            f"val_domain_auc={auc_domain_val:.5f}",
            flush=True,
        )

        if auc_final_val > best_auc:
            best_auc = auc_final_val
            model.save(best_model_path)
            print(f"{tag} fold {fold}: saved best model with val AUC {best_auc:.5f}", flush=True)

    with open(f"{output_dir}/models/history_{tag}_fold_{fold}.pkl", "wb") as f:
        pickle.dump(history, f)

    # Also keep old style history name if only one mode is trained
    if SAVE_MODE == "domain" and use_domain:
        with open(f"{output_dir}/models/history_fold_{fold}.pkl", "wb") as f:
            pickle.dump(history, f)
    if SAVE_MODE == "nodomain" and not use_domain:
        with open(f"{output_dir}/models/history_fold_{fold}.pkl", "wb") as f:
            pickle.dump(history, f)

    plot_training_history(history, fold, tag)

    custom_objects = {
        "GradientReversalLayer": GradientReversalLayer,
        "gradient_reverse": gradient_reverse,
    }

    best_model = tf.keras.models.load_model(
        best_model_path,
        custom_objects=custom_objects,
        compile=False,
    )

    print(f"{tag} fold {fold}: start final test evaluation", flush=True)

    tk_test_m, tk_test_t = arrays_test["task_X"]
    task_test = dfs_val_test["task_test"]

    test_outputs = predict_outputs(best_model, tk_test_m, tk_test_t, use_domain=use_domain)

    step1_score = test_outputs[0]
    step2_score = test_outputs[1]
    step3_score = test_outputs[2]
    step4_score = test_outputs[3]
    final_score = test_outputs[4]

    y_task_test = task_test["task_label"].values
    w_task_test_train = task_test["train_weight"].values
    w_task_test_phys = task_test["signed_weight"].values

    final_test_auc = safe_auc(y_task_test, final_score, w_task_test_train)

    if use_domain:
        dm_test_m, dm_test_t = arrays_test["domain_X"]
        domain_test = dfs_val_test["domain_test"]

        domain_outputs = predict_outputs(best_model, dm_test_m, dm_test_t, use_domain=True)
        domain_task_score = domain_outputs[4]
        domain_score = domain_outputs[5]

        y_domain_test = domain_test["domain_label"].values
        w_domain_test_train = domain_test["train_weight"].values
        w_domain_test_phys = domain_test["signed_weight"].values

        domain_test_auc = safe_auc(y_domain_test, domain_score, w_domain_test_train)
    else:
        domain_task_score = None
        domain_score = None
        y_domain_test = None
        w_domain_test_phys = None
        domain_test_auc = np.nan

    print(f"[{tag} fold {fold}] final task test AUC = {final_test_auc:.6f}", flush=True)
    print(f"[{tag} fold {fold}] domain test AUC     = {domain_test_auc:.6f}", flush=True)

    np.save(f"{output_dir}/arrays/{tag}_step1_score_fold_{fold}.npy", step1_score)
    np.save(f"{output_dir}/arrays/{tag}_step2_score_fold_{fold}.npy", step2_score)
    np.save(f"{output_dir}/arrays/{tag}_step3_score_fold_{fold}.npy", step3_score)
    np.save(f"{output_dir}/arrays/{tag}_step4_score_fold_{fold}.npy", step4_score)
    np.save(f"{output_dir}/arrays/{tag}_final_score_fold_{fold}.npy", final_score)
    np.save(f"{output_dir}/arrays/{tag}_task_label_fold_{fold}.npy", y_task_test)
    np.save(f"{output_dir}/arrays/{tag}_task_weight_fold_{fold}.npy", w_task_test_phys)

    if use_domain:
        np.save(f"{output_dir}/arrays/{tag}_domain_score_fold_{fold}.npy", domain_score)
        np.save(f"{output_dir}/arrays/{tag}_domain_label_fold_{fold}.npy", y_domain_test)
        np.save(f"{output_dir}/arrays/{tag}_domain_task_score_fold_{fold}.npy", domain_task_score)

    plot_roc(
        y_task_test,
        final_score,
        w_task_test_train,
        f"{tag} final task ROC, fold {fold}",
        f"{output_dir}/plots/{tag}_final_task_roc_fold_{fold}.png",
    )

    plot_score_distribution(
        y_task_test,
        final_score,
        w_task_test_phys,
        f"{tag} final task score distribution, fold {fold}",
        f"{output_dir}/plots/{tag}_final_task_score_dist_fold_{fold}.png",
        names=("background MC", "signal MC"),
    )

    plot_score_vs_mass(
        task_test,
        final_score,
        f"{tag} final DNN score vs m_mumu, fold {fold}",
        f"{output_dir}/plots/{tag}_final_score_vs_mass_fold_{fold}.png",
    )

    if use_domain:
        domain_test = dfs_val_test["domain_test"]

        plot_roc(
            y_domain_test,
            domain_score,
            w_domain_test_train,
            f"{tag} domain ROC, fold {fold}",
            f"{output_dir}/plots/{tag}_domain_roc_fold_{fold}.png",
        )

        plot_score_distribution(
            y_domain_test,
            domain_score,
            w_domain_test_phys,
            f"{tag} domain score distribution, fold {fold}",
            f"{output_dir}/plots/{tag}_domain_score_dist_fold_{fold}.png",
            names=("background MC SB", "data SB"),
        )

        plot_sb_data_mc_ratio(
            domain_test,
            domain_task_score,
            f"{output_dir}/plots/{tag}_sb_data_mc_ratio_vs_final_score_fold_{fold}.png",
            n_bins=10,
        )

    # DNN_t transformer is fit on final test score.
    # This matches your old fold-wise application style.
    tsf = QuantileTransformer(
        n_quantiles=min(1000, len(final_score)),
        output_distribution="uniform",
        subsample=1000000000,
        random_state=0,
    )
    tsf.fit(final_score.reshape(-1, 1))

    if use_domain:
        tsf_name = f"DNN_tsf_DA_fold_{fold}"
    else:
        tsf_name = f"DNN_tsf_noDomain_fold_{fold}"

    with open(f"{output_dir}/models/{tsf_name}.pkl", "wb") as f:
        pickle.dump(tsf, f, protocol=-1)

    joblib.dump(tsf, f"{output_dir}/models/{tsf_name}.joblib")

    # Compatibility names if only one mode
    if SAVE_MODE == "domain" and use_domain:
        with open(f"{output_dir}/models/DNN_tsf_fold_{fold}.pkl", "wb") as f:
            pickle.dump(tsf, f, protocol=-1)
        joblib.dump(tsf, f"{output_dir}/models/DNN_tsf_fold_{fold}.joblib")

    if SAVE_MODE == "nodomain" and not use_domain:
        with open(f"{output_dir}/models/DNN_tsf_fold_{fold}.pkl", "wb") as f:
            pickle.dump(tsf, f, protocol=-1)
        joblib.dump(tsf, f"{output_dir}/models/DNN_tsf_fold_{fold}.joblib")

    summary = {
        "tag": tag,
        "fold": fold,
        "use_domain": use_domain,
        "final_task_test_auc": float(final_test_auc),
        "domain_test_auc": float(domain_test_auc),
        "best_val_auc": float(best_auc),
        "model_path": best_model_path,
        "tsf_path": f"{output_dir}/models/{tsf_name}.pkl",
    }

    del model
    del best_model
    tf.keras.backend.clear_session()
    gc.collect()

    return summary


# ============================================================
# Main training loop
# ============================================================

fold_summaries = []

for fold in range(1, n_folds + 1):
    print("\n" + "=" * 90, flush=True)
    print(f"Training fold {fold}", flush=True)
    print("=" * 90, flush=True)

    step1_train, step1_val, step1_test = split_by_fold(step1_df, fold, "label_step1")
    step2_train, step2_val, step2_test = split_by_fold(step2_df, fold, "label_step2")
    task_train, task_val, task_test = split_by_fold(task_df, fold, "task_label")
    domain_train, domain_val, domain_test = split_by_fold(domain_df, fold, "domain_label")

    print(f"Fold {fold}: step1 train/val/test = {len(step1_train)}, {len(step1_val)}, {len(step1_test)}", flush=True)
    print(f"Fold {fold}: step2 train/val/test = {len(step2_train)}, {len(step2_val)}, {len(step2_test)}", flush=True)
    print(f"Fold {fold}: task train/val/test = {len(task_train)}, {len(task_val)}, {len(task_test)}", flush=True)
    print(f"Fold {fold}: domain train/val/test = {len(domain_train)}, {len(domain_val)}, {len(domain_test)}", flush=True)

    scaler = fit_scaler_from_task_train(task_train)

    # Common scaler for both DA and no-domain
    joblib.dump(scaler, f"{output_dir}/models/scaler_fold_{fold}.pkl")

    # Task arrays, original mass
    s1_train_m, s1_train_t = transform_df(step1_train, scaler, fix_domain_mass=False)
    s1_val_m, s1_val_t = transform_df(step1_val, scaler, fix_domain_mass=False)
    s1_test_m, s1_test_t = transform_df(step1_test, scaler, fix_domain_mass=False)

    s2_train_m, s2_train_t = transform_df(step2_train, scaler, fix_domain_mass=False)
    s2_val_m, s2_val_t = transform_df(step2_val, scaler, fix_domain_mass=False)
    s2_test_m, s2_test_t = transform_df(step2_test, scaler, fix_domain_mass=False)

    tk_train_m, tk_train_t = transform_df(task_train, scaler, fix_domain_mass=False)
    tk_val_m, tk_val_t = transform_df(task_val, scaler, fix_domain_mass=False)
    tk_test_m, tk_test_t = transform_df(task_test, scaler, fix_domain_mass=False)

    # Domain arrays, SB mass fixed to 125
    dm_train_m, dm_train_t = transform_df(domain_train, scaler, fix_domain_mass=True)
    dm_val_m, dm_val_t = transform_df(domain_val, scaler, fix_domain_mass=True)
    dm_test_m, dm_test_t = transform_df(domain_test, scaler, fix_domain_mass=True)

    arrays_train_common = {
        "step1": (
            s1_train_m,
            s1_train_t,
            step1_train["label_step1"].values,
            step1_train["train_weight"].values,
        ),
        "step2": (
            s2_train_m,
            s2_train_t,
            step2_train["label_step2"].values,
            step2_train["train_weight"].values,
        ),
        "task": (
            tk_train_m,
            tk_train_t,
            task_train["task_label"].values,
            task_train["train_weight"].values,
        ),
        "domain": (
            dm_train_m,
            dm_train_t,
            domain_train["domain_label"].values,
            domain_train["train_weight"].values,
        ),
    }

    arrays_val_common = {
        "task_X": (
            tk_val_m,
            tk_val_t,
        ),
        "domain_X": (
            dm_val_m,
            dm_val_t,
        ),
    }

    arrays_test_common = {
        "task_X": (
            tk_test_m,
            tk_test_t,
        ),
        "domain_X": (
            dm_test_m,
            dm_test_t,
        ),
    }

    dfs_val_test_common = {
        "task_val": task_val,
        "task_test": task_test,
        "domain_val": domain_val,
        "domain_test": domain_test,
    }

    if TRAIN_NODOMAIN:
        summary_no = train_variant_for_fold(
            fold=fold,
            tag="noDomain",
            use_domain=False,
            scaler=scaler,
            arrays_train=arrays_train_common,
            arrays_val=arrays_val_common,
            arrays_test=arrays_test_common,
            dfs_val_test=dfs_val_test_common,
        )
        fold_summaries.append(summary_no)

        with open(f"{output_dir}/fold_summary.json", "w") as f:
            json.dump(to_jsonable(fold_summaries), f, indent=2)

    if TRAIN_DOMAIN:
        summary_da = train_variant_for_fold(
            fold=fold,
            tag="DA",
            use_domain=True,
            scaler=scaler,
            arrays_train=arrays_train_common,
            arrays_val=arrays_val_common,
            arrays_test=arrays_test_common,
            dfs_val_test=dfs_val_test_common,
        )
        fold_summaries.append(summary_da)

        with open(f"{output_dir}/fold_summary.json", "w") as f:
            json.dump(to_jsonable(fold_summaries), f, indent=2)

    print(f"Fold {fold}: finished", flush=True)

    gc.collect()


# ============================================================
# Run info
# ============================================================

run_info = {
    "output_dir": output_dir,
    "SAVE_MODE": SAVE_MODE,
    "TRAIN_NODOMAIN": TRAIN_NODOMAIN,
    "TRAIN_DOMAIN": TRAIN_DOMAIN,
    "ONLY_YEAR": ONLY_YEAR,
    "n_folds": n_folds,
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "dropout": dropout,
    "alpha": alpha,
    "max_lambda": max_lambda,
    "focus_weight_max": focus_weight_max,
    "warmup_epochs": warmup_epochs,
    "FIX_DOMAIN_MASS_TO_125": FIX_DOMAIN_MASS_TO_125,
    "DOMAIN_FIXED_MASS_VALUE": DOMAIN_FIXED_MASS_VALUE,
    "weight_treatment": "signed physical weights for task and domain; BCE normalized by sum(abs(weight))",
    "branch_inputs": {
        "step1": "mass_res + vbf_topo",
        "step2": "mass_res + vbf_topo",
        "step3": "vbf_topo only",
        "step4": "mass_res only",
    },
    "fold_summaries": fold_summaries,
}

with open(f"{output_dir}/run_info.json", "w") as f:
    json.dump(to_jsonable(run_info), f, indent=2)

with open(f"{output_dir}/run_info.txt", "w") as f:
    f.write(json.dumps(to_jsonable(run_info), indent=2))

print("\nTraining finished.", flush=True)
print("Output:", output_dir, flush=True)
print(json.dumps(to_jsonable(fold_summaries), indent=2), flush=True)
