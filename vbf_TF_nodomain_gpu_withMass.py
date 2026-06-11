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
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import roc_curve, auc, roc_auc_score


# ============================================================
# GPU / multi-core setup
# ============================================================

NCPU = int(os.environ.get("_CONDOR_NPROCS", os.environ.get("OMP_NUM_THREADS", "4")))

os.environ["OMP_NUM_THREADS"] = str(NCPU)
os.environ["MKL_NUM_THREADS"] = str(NCPU)
os.environ["OPENBLAS_NUM_THREADS"] = str(NCPU)
os.environ["NUMEXPR_NUM_THREADS"] = str(NCPU)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(NCPU)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

tf.config.threading.set_intra_op_parallelism_threads(NCPU)
tf.config.threading.set_inter_op_parallelism_threads(2)

gpus = tf.config.list_physical_devices("GPU")
print("[INFO] Available GPUs:", gpus, flush=True)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] GPU memory growth enabled.", flush=True)
    except RuntimeError as e:
        print("[WARNING] Could not set GPU memory growth:", e, flush=True)

USE_MIXED_PRECISION = True

if USE_MIXED_PRECISION and gpus:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    print("[INFO] Mixed precision enabled: mixed_float16", flush=True)
else:
    print("[INFO] Mixed precision disabled.", flush=True)


# ============================================================
# Configuration
# ============================================================

path = {}

path[0] = "/eos/user/h/hakou/Hmumu_Share/qguo/2025/skimmed_ntuples/2022/SRSB_noJetHornVeto"
path[1] = "/eos/user/h/hakou/Hmumu_Share/qguo/2025/skimmed_ntuples/2022EE/SRSB_noJetHornVeto"
path[2] = "/eos/user/h/hakou/Hmumu_Share/qguo/2025/skimmed_ntuples/2023/SRSB_noJetHornVeto"
path[3] = "/eos/user/h/hakou/Hmumu_Share/qguo/2025/skimmed_ntuples/2023BPix/SRSB_noJetHornVeto"
path[4] = "/eos/user/h/hakou/Hmumu_Share/qguo/2025/skimmed_ntuples/2024/SRSB_noJetHornVeto"
path[5] = "/eos/user/h/hakou/Hmumu_Share/qguo/2025/skimmed_ntuples/2025/SRSB_noJetHornVeto"

era_names = {
    0: "2022",
    1: "2022EE",
    2: "2023",
    3: "2023BPix",
    4: "2024",
    5: "2025",
}

tree_name_by_path = {
    0: "v0",
    1: "v2",
    2: "data_two_jet_m110To150",
    3: "data_two_jet_m110To150",
    4: "data_two_jet_m110To150",
    5: "data_two_jet_m110To150",
}

signal_files_by_path = {
    0: {"VBFHToMuMu_M125.root": "SIGNAL"},
    1: {"VBFHToMuMu_M125.root": "SIGNAL"},
    2: {"VBFHToMuMu_M125.root": "SIGNAL"},
    3: {"VBFHToMuMu_M125.root": "SIGNAL"},
    4: {"VBFHToMuMu_M125.root": "SIGNAL"},
    5: {"VBFHToMuMu_M125.root": "SIGNAL"},
}

background_files_by_path = {
    0: {
        "DY_105To160_Inc_failvbffilter.root": "DY",
        "DY_105To160_Fil-VBF_passvbffilter.root": "DY",
        "EWK_LLJJ_M105To160.root": "EWK_ZJJ",
        "TTTo2L2Nu.root": "TT",
    },
    1: {
        "DY_105To160_Inc_failvbffilter.root": "DY",
        "DY_105To160_Fil-VBF_passvbffilter.root": "DY",
        "EWK_LLJJ_M105To160.root": "EWK_ZJJ",
        "TTTo2L2Nu.root": "TT",
    },
    2: {
        "DY_105To160_Inc_failvbffilter.root": "DY",
        "DY_105To160_Fil-VBF_passvbffilter.root": "DY",
        "EWK_LLJJ_M105To160.root": "EWK_ZJJ",
        "TTTo2L2Nu.root": "TT",
    },
    3: {
        "DY_105To160_Inc_failvbffilter.root": "DY",
        "DY_105To160_Fil-VBF_passvbffilter.root": "DY",
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
    0: {"Data_2022.root": "DATA"},
    1: {"Data_2022EE.root": "DATA"},
    2: {"Data_2023.root": "DATA"},
    3: {"Data_2023BPix.root": "DATA"},
    4: {"Data_2024.root": "DATA"},
    5: {"Data_2025.root": "DATA"},
}


# ============================================================
# Quick switch
# ============================================================

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
# Variables
# ============================================================

target_cate_index = 3

mass_branch = "diMufsr_kit_BSC_mass"
weight_branch = "eventWeight"

sr_low = 115.0
sr_high = 135.0

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
]

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

mass_like_features = [
    "diMufsr_kit_BSC_mass",
    "diMu-mass_resolution_abs",
    "diMu-mass_resolution",
]

transformer_features_base = feature_branches.copy()


# ============================================================
# Training hyperparameters
# ============================================================

n_folds = 4

epochs = 60
#batch_size = 8192
batch_size = 4096
learning_rate = 8e-4
weight_decay = 1e-5

validate_every = 1
early_stop_patience = 10

# FT-Transformer hyperparameters
d_token = 32
n_heads = 4
n_transformer_blocks = 3
ff_dim = 96
dropout = 0.15


# ============================================================
# Output
# ============================================================

current_date = datetime.now().strftime("%m%d")

path_out = "/eos/user/q/qguo/SWAN_projects/ML_test"
output_name = f"saved_model_noDomain_FTTransformer_withMass_GPU_2026{current_date}_v1"
output_dir = os.path.join(path_out, output_name)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "arrays"), exist_ok=True)

print("[INFO] output_dir =", output_dir, flush=True)

with open(os.path.join(output_dir, "run_info.txt"), "w") as f:
    f.write(f"hostname = {os.popen('hostname').read()}\n")
    f.write(f"date = {os.popen('date').read()}\n")
    f.write(f"output_dir = {output_dir}\n")
    f.write("training_type = no_domain_ft_transformer_with_mass_gpu\n")
    f.write(f"NCPU = {NCPU}\n")
    f.write(f"gpus = {gpus}\n")
    f.write(f"mixed_precision = {USE_MIXED_PRECISION}\n")


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
        raise RuntimeError("No file records were built.")

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
        raise RuntimeError("No files loaded.")

    return pd.concat(dfs, ignore_index=True)


records = build_file_records()
df = load_data_from_records(records, branches)

print("[INFO] Loaded events:", len(df), flush=True)


# ============================================================
# Selection and cleaning
# ============================================================

if "cate_index" in df.columns:
    df = df[df["cate_index"] == target_cate_index].copy()
    print(f"[INFO] After cate_index == {target_cate_index}: {len(df)}", flush=True)

if weight_branch not in df.columns:
    raise RuntimeError(f"Missing weight branch: {weight_branch}")

df["weight"] = df[weight_branch].astype(float)
df.loc[df["sample_type"] == "data", "weight"] = 1.0

df["weight"] = df["weight"].replace([np.inf, -np.inf], np.nan)
df["weight"] = df["weight"].fillna(0.0)

df["signed_weight"] = df["weight"]
df["abs_weight"] = np.abs(df["weight"])

df["is_sr"] = ((df[mass_branch] > sr_low) & (df[mass_branch] < sr_high)).astype(int)
df["is_sb"] = ((df[mass_branch] < sr_low) | (df[mass_branch] > sr_high)).astype(int)

for era in era_names.values():
    df[f"era_{era}"] = (df["era"] == era).astype(float)

year_features = [f"era_{era}" for era in era_names.values()]
transformer_feature_branches = transformer_features_base + year_features

for col in transformer_feature_branches:
    if col not in df.columns:
        raise RuntimeError(f"Missing transformer feature branch: {col}")

df[transformer_feature_branches] = df[transformer_feature_branches].replace([np.inf, -np.inf], np.nan)
df[transformer_feature_branches] = df[transformer_feature_branches].fillna(-1.0)

with open(os.path.join(output_dir, "features.json"), "w") as f:
    json.dump({
        "feature_branches": feature_branches,
        "mass_like_features_included": mass_like_features,
        "transformer_features_base": transformer_features_base,
        "year_features": year_features,
        "all_feature_branches": transformer_feature_branches,
        "model_input_type": "mass_plus_topology_ft_transformer_gpu",
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
        "training_type": "no_domain_ft_transformer_with_mass_gpu",
    }, f, indent=2)


# ============================================================
# Dataset construction
# ============================================================

task_df = df[
    (df["is_sr"] == 1)
    & (df["sample_type"].isin(["signal", "background"]))
].copy()
task_df["task_label"] = (task_df["sample_type"] == "signal").astype(float)

domain_df = df[
    (df["is_sb"] == 1)
    & (df["sample_type"].isin(["data", "background"]))
].copy()
domain_df["domain_label"] = (domain_df["sample_type"] == "data").astype(float)

if len(task_df) == 0:
    raise RuntimeError("task_df is empty.")

print("\n[Dataset sizes]", flush=True)
print("Task signal vs all background:", len(task_df), flush=True)
print(task_df.groupby(["sample_type", "process"])["weight"].sum(), flush=True)

print("\nSB Data/MC validation sample:", len(domain_df), flush=True)
if len(domain_df) > 0:
    print(domain_df.groupby(["sample_type", "process"])["weight"].sum(), flush=True)


# ============================================================
# Training weights
# ============================================================

def add_train_weight(df_in, label_col):
    out = df_in.copy()

    out["train_weight"] = out["abs_weight"].astype(float)

    labels = sorted(out[label_col].unique())
    n_class = len(labels)

    for label in labels:
        mask = out[label_col] == label
        sumw = out.loc[mask, "train_weight"].sum()
        if sumw > 0:
            out.loc[mask, "train_weight"] *= len(out) / (n_class * sumw)

    out["train_weight"] = out["train_weight"].replace([np.inf, -np.inf], np.nan)
    out["train_weight"] = out["train_weight"].fillna(0.0)

    return out


task_df = add_train_weight(task_df, "task_label")

if len(domain_df) > 0:
    domain_df["train_weight"] = domain_df["abs_weight"]


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


task_df = assign_folds(task_df, n_folds=n_folds, random_state=33)

if len(domain_df) > 0:
    domain_df = assign_folds(domain_df, n_folds=n_folds, random_state=44)

task_df.to_pickle(os.path.join(output_dir, "task_df_with_folds.pkl"))
domain_df.to_pickle(os.path.join(output_dir, "domain_df_with_folds.pkl"))


# ============================================================
# Custom serializable layers
# ============================================================

@tf.keras.utils.register_keras_serializable(package="VBF")
class FeatureTokenizer(tf.keras.layers.Layer):
    def __init__(self, n_features, d_token, **kwargs):
        super().__init__(**kwargs)
        self.n_features = int(n_features)
        self.d_token = int(d_token)

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="feature_weight",
            shape=(self.n_features, self.d_token),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="feature_bias",
            shape=(self.n_features, self.d_token),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        x = tf.expand_dims(x, axis=-1)
        return x * self.weight + self.bias

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "n_features": self.n_features,
            "d_token": self.d_token,
        })
        return cfg


@tf.keras.utils.register_keras_serializable(package="VBF")
class CLSAppend(tf.keras.layers.Layer):
    def __init__(self, d_token, **kwargs):
        super().__init__(**kwargs)
        self.d_token = int(d_token)

    def build(self, input_shape):
        self.cls = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.d_token),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        batch = tf.shape(x)[0]
        cls = tf.tile(self.cls, [batch, 1, 1])
        return tf.concat([cls, x], axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_token": self.d_token})
        return cfg


@tf.keras.utils.register_keras_serializable(package="VBF")
class TakeCLSToken(tf.keras.layers.Layer):
    def call(self, x):
        return x[:, 0, :]

    def get_config(self):
        return super().get_config()


# ============================================================
# Model
# ============================================================

def transformer_block(x, d_token, n_heads, ff_dim, dropout, name):
    x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_attn_ln")(x)

    attn = tf.keras.layers.MultiHeadAttention(
        num_heads=n_heads,
        key_dim=d_token // n_heads,
        dropout=dropout,
        name=f"{name}_mha",
    )(x_norm, x_norm)

    attn = tf.keras.layers.Dropout(dropout, name=f"{name}_attn_dropout")(attn)
    x = tf.keras.layers.Add(name=f"{name}_attn_add")([x, attn])

    x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ffn_ln")(x)

    ff = tf.keras.layers.Dense(ff_dim, activation="gelu", name=f"{name}_ffn_dense1")(x_norm)
    ff = tf.keras.layers.Dropout(dropout, name=f"{name}_ffn_dropout1")(ff)
    ff = tf.keras.layers.Dense(d_token, activation=None, name=f"{name}_ffn_dense2")(ff)
    ff = tf.keras.layers.Dropout(dropout, name=f"{name}_ffn_dropout2")(ff)

    x = tf.keras.layers.Add(name=f"{name}_ffn_add")([x, ff])

    return x


def build_ft_transformer_model(n_features):
    inputs = tf.keras.layers.Input(shape=(n_features,), name="features")

    tokens = FeatureTokenizer(
        n_features=n_features,
        d_token=d_token,
        name="feature_tokenizer",
    )(inputs)

    tokens = CLSAppend(
        d_token=d_token,
        name="append_cls",
    )(tokens)

    x = tokens

    for iblock in range(n_transformer_blocks):
        x = transformer_block(
            x,
            d_token=d_token,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            name=f"transformer_block_{iblock + 1}",
        )

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)

    cls = TakeCLSToken(name="take_cls")(x)

    head = tf.keras.layers.Dense(64, activation="gelu", name="head_dense1")(cls)
    head = tf.keras.layers.Dropout(dropout, name="head_dropout1")(head)
    head = tf.keras.layers.Dense(32, activation="gelu", name="head_dense2")(head)
    head = tf.keras.layers.Dropout(dropout, name="head_dropout2")(head)

    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        dtype="float32",
        name="Final_Task_Output",
    )(head)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=output,
        name="NoDomain_FTTransformer_WithMass_GPU",
    )

    return model


# ============================================================
# Array helpers
# ============================================================

def fit_scaler_from_task_train(task_train):
    scaler = StandardScaler()
    X = task_train[transformer_feature_branches].values
    scaler.fit(X)
    return scaler


def transform_df(df_in, scaler):
    X = df_in[transformer_feature_branches].values
    X_scaled = scaler.transform(X)
    return X_scaled.astype(np.float32)


def make_tf_dataset(X, y, w, batch_size, training=True):
    y = y.astype(np.float32).reshape(-1, 1)
    w = w.astype(np.float32).reshape(-1)

    ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y, w))

    if training:
        buffer_size = min(len(y), 200000)
        ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def predict_score(model, X, batch_size=16384):
    ds = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    score = model.predict(ds, verbose=0).reshape(-1)
    return score


# ============================================================
# Plot helpers
# ============================================================

def safe_auc(y, score, w):
    try:
        return roc_auc_score(y, score, sample_weight=w)
    except Exception:
        return np.nan


def plot_roc(y, score, w, title, outpath):
    fpr, tpr, _ = roc_curve(y, score, sample_weight=w)
    roc_auc = auc(fpr, tpr)

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


def plot_bkg_rejection(y, score, w, title, outpath):
    fpr, tpr, _ = roc_curve(y, score, sample_weight=w)

    eps_sig = tpr
    eps_bkg = fpr
    bkg_rej = 1.0 - eps_bkg

    plt.figure(figsize=(7, 6))
    plt.plot(eps_sig, bkg_rej)
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background rejection")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_score_distribution(y, score, w, title, outpath, xlabel="Transformer score"):
    plt.figure(figsize=(8, 6))

    for label, name in [(0, "background MC"), (1, "signal MC")]:
        mask = y == label
        plt.hist(
            score[mask],
            bins=50,
            range=(0, 1),
            weights=w[mask],
            density=True,
            histtype="step",
            linewidth=1.5,
            label=name,
        )

    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_training_history(history, fold):
    hist = history.history
    epochs_arr = np.arange(1, len(hist["loss"]) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_arr, hist["loss"], label="train loss")
    if "val_loss" in hist:
        plt.plot(epochs_arr, hist["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted BCE loss")
    plt.title(f"FT-Transformer with mass loss, fold {fold}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", f"loss_fold_{fold}.png"))
    plt.close()

    auc_key = "auc"
    val_auc_key = "val_auc"

    plt.figure(figsize=(8, 6))
    if auc_key in hist:
        plt.plot(epochs_arr, hist[auc_key], label="train AUC")
    if val_auc_key in hist:
        plt.plot(epochs_arr, hist[val_auc_key], label="val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(f"FT-Transformer with mass AUC, fold {fold}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", f"auc_history_fold_{fold}.png"))
    plt.close()


def plot_score_vs_mass(df_eval, score, title, outpath):
    plt.figure(figsize=(8, 6))
    plt.hist2d(df_eval[mass_branch].values, score, bins=(60, 60))
    plt.xlabel(r"$m_{\mu\mu}$")
    plt.ylabel("Transformer score")
    plt.title(title)
    plt.colorbar(label="Events")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def compute_data_mc_ratio(df_eval, score, n_bins=10, weight_col="weight"):
    tmp = df_eval.copy()
    tmp["score"] = score

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    data_y = []
    mc_y = []
    data_err = []
    mc_err = []

    for ibin, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if ibin == len(bins) - 2:
            in_bin = (tmp["score"] >= lo) & (tmp["score"] <= hi)
        else:
            in_bin = (tmp["score"] >= lo) & (tmp["score"] < hi)

        data_mask = in_bin & (tmp["sample_type"] == "data")
        mc_mask = in_bin & (tmp["sample_type"] == "background")

        data_w = tmp.loc[data_mask, weight_col].values.astype(float)
        mc_w = tmp.loc[mc_mask, weight_col].values.astype(float)

        data_y.append(np.sum(data_w))
        mc_y.append(np.sum(mc_w))
        data_err.append(np.sqrt(np.sum(data_w ** 2)))
        mc_err.append(np.sqrt(np.sum(mc_w ** 2)))

    data_y = np.array(data_y)
    mc_y = np.array(mc_y)
    data_err = np.array(data_err)
    mc_err = np.array(mc_err)

    scale = 1.0
    if np.sum(mc_y) != 0:
        scale = np.sum(data_y) / np.sum(mc_y)

    mc_y_scaled = mc_y * scale
    mc_err_scaled = mc_err * scale

    ratio = np.full_like(data_y, np.nan, dtype=float)
    ratio_err = np.full_like(data_y, np.nan, dtype=float)

    valid = mc_y_scaled != 0
    ratio[valid] = data_y[valid] / mc_y_scaled[valid]

    good_den = valid & (np.abs(data_y) > 0) & (np.abs(mc_y_scaled) > 0)

    ratio_err[good_den] = ratio[good_den] * np.sqrt(
        (data_err[good_den] / np.abs(data_y[good_den])) ** 2
        + (mc_err_scaled[good_den] / np.abs(mc_y_scaled[good_den])) ** 2
    )

    good = good_den & np.isfinite(ratio) & np.isfinite(ratio_err) & (ratio_err > 0)

    if np.sum(good) > 1:
        chi2 = np.sum(((ratio[good] - 1.0) / ratio_err[good]) ** 2)
        ndof = np.sum(good) - 1
        chi2_ndof = chi2 / ndof
    else:
        chi2 = np.nan
        ndof = 0
        chi2_ndof = np.nan

    return {
        "bins": bins,
        "centers": centers,
        "data_y": data_y,
        "data_err": data_err,
        "mc_y": mc_y_scaled,
        "mc_err": mc_err_scaled,
        "ratio": ratio,
        "ratio_err": ratio_err,
        "chi2": chi2,
        "ndof": ndof,
        "chi2_ndof": chi2_ndof,
    }


def plot_sb_data_mc_ratio(domain_eval_df, score, outpath, score_name="Transformer score", n_bins=10):
    result = compute_data_mc_ratio(domain_eval_df, score, n_bins=n_bins, weight_col="weight")

    bins = result["bins"]
    centers = result["centers"]

    plt.figure(figsize=(8, 7))

    ax1 = plt.subplot(2, 1, 1)
    ax1.step(
        bins,
        np.r_[result["mc_y"], result["mc_y"][-1]],
        where="post",
        label="MC, normalized to Data",
    )
    ax1.errorbar(
        centers,
        result["data_y"],
        yerr=result["data_err"],
        fmt="o",
        color="black",
        label="Data",
    )
    ax1.set_yscale("log")
    ax1.set_ylabel("Events")
    ax1.set_title(f"SB Data/MC vs {score_name}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.errorbar(
        centers,
        result["ratio"],
        yerr=result["ratio_err"],
        fmt="o",
        label=f"chi2/ndof = {result['chi2_ndof']:.3f}",
    )
    ax2.axhline(1.0, linestyle="--", color="black")
    ax2.set_xlabel(score_name)
    ax2.set_ylabel("Data / MC")
    ax2.set_ylim(0.75, 1.25)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

    return result


# ============================================================
# Main 4-fold loop
# ============================================================

fold_summaries = []

for fold in range(1, n_folds + 1):
    print("\n" + "=" * 90, flush=True)
    print(f"FT-Transformer with mass no-domain GPU training fold {fold}", flush=True)
    print("=" * 90, flush=True)

    task_train, task_val, task_test = split_by_fold(task_df, fold, "task_label")

    if len(domain_df) > 0:
        domain_test = domain_df[domain_df["fold_id"] == fold].copy()
    else:
        domain_test = pd.DataFrame()

    print(f"[INFO] Fold {fold}: task train/val/test = {len(task_train)}, {len(task_val)}, {len(task_test)}", flush=True)
    print(f"[INFO] Fold {fold}: SB validation events = {len(domain_test)}", flush=True)

    scaler = fit_scaler_from_task_train(task_train)
    joblib.dump(scaler, os.path.join(output_dir, "models", f"scaler_fold_{fold}.pkl"))

    X_train = transform_df(task_train, scaler)
    X_val = transform_df(task_val, scaler)
    X_test = transform_df(task_test, scaler)

    y_train = task_train["task_label"].values.astype(np.float32)
    w_train = task_train["train_weight"].values.astype(np.float32)

    y_val = task_val["task_label"].values.astype(np.float32)
    w_val = task_val["train_weight"].values.astype(np.float32)

    y_test = task_test["task_label"].values.astype(np.float32)
    w_test_train = task_test["train_weight"].values.astype(np.float32)
    w_test_phys = task_test["weight"].values.astype(np.float32)

    train_ds = make_tf_dataset(X_train, y_train, w_train, batch_size=batch_size, training=True)
    val_ds = make_tf_dataset(X_val, y_val, w_val, batch_size=batch_size, training=False)

    model = build_ft_transformer_model(n_features=len(transformer_feature_branches))

    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    except Exception:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.BinaryAccuracy(name="acc"),
        ],
    )

    model.summary(print_fn=lambda x: print(x, flush=True))

    best_model_path = os.path.join(
        output_dir,
        "models",
        f"transformer_noDomain_withMass_GPU_model_fold_{fold}.keras",
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=early_stop_patience,
            restore_best_weights=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, "models", f"training_log_fold_{fold}.csv")
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )

    with open(os.path.join(output_dir, "models", f"history_fold_{fold}.pkl"), "wb") as f:
        pickle.dump(history.history, f)

    plot_training_history(history, fold)

    best_model = tf.keras.models.load_model(
        best_model_path,
        custom_objects={
            "FeatureTokenizer": FeatureTokenizer,
            "CLSAppend": CLSAppend,
            "TakeCLSToken": TakeCLSToken,
        },
        compile=False,
    )

    print(f"[INFO] Fold {fold}: start final test evaluation", flush=True)

    final_score = predict_score(best_model, X_test)

    final_test_auc = safe_auc(y_test, final_score, w_test_train)

    print(f"[RESULT] Fold {fold}: final task test AUC = {final_test_auc:.6f}", flush=True)

    np.save(os.path.join(output_dir, "arrays", f"final_score_fold_{fold}.npy"), final_score)
    np.save(os.path.join(output_dir, "arrays", f"task_label_fold_{fold}.npy"), y_test)
    np.save(os.path.join(output_dir, "arrays", f"task_weight_fold_{fold}.npy"), w_test_phys)
    np.save(os.path.join(output_dir, "arrays", f"task_train_weight_fold_{fold}.npy"), w_test_train)

    plot_roc(
        y_test,
        final_score,
        w_test_train,
        f"FT-Transformer with mass ROC, fold {fold}",
        os.path.join(output_dir, "plots", f"final_task_roc_fold_{fold}.png"),
    )

    plot_bkg_rejection(
        y_test,
        final_score,
        w_test_train,
        f"FT-Transformer with mass background rejection, fold {fold}",
        os.path.join(output_dir, "plots", f"bkg_rejection_fold_{fold}.png"),
    )

    plot_score_distribution(
        y_test,
        final_score,
        w_test_phys,
        f"FT-Transformer with mass score distribution, fold {fold}",
        os.path.join(output_dir, "plots", f"final_task_score_dist_fold_{fold}.png"),
        xlabel="Transformer score",
    )

    plot_score_vs_mass(
        task_test,
        final_score,
        f"FT-Transformer with mass score vs m_mumu, fold {fold}",
        os.path.join(output_dir, "plots", f"final_score_vs_mass_fold_{fold}.png"),
    )

    # DNN_t / Transformer_t
    tsf = QuantileTransformer(
        n_quantiles=min(1000, len(final_score)),
        output_distribution="uniform",
        subsample=1000000000,
        random_state=0,
    )
    tsf.fit(final_score.reshape(-1, 1))

    with open(os.path.join(output_dir, "models", f"DNN_tsf_fold_{fold}.pkl"), "wb") as f:
        pickle.dump(tsf, f, protocol=-1)

    joblib.dump(tsf, os.path.join(output_dir, "models", f"DNN_tsf_fold_{fold}.joblib"))

    final_score_t = tsf.transform(final_score.reshape(-1, 1)).reshape(-1)
    np.save(os.path.join(output_dir, "arrays", f"final_score_t_fold_{fold}.npy"), final_score_t)

    plot_score_distribution(
        y_test,
        final_score_t,
        w_test_phys,
        f"FT-Transformer with mass transformed score distribution, fold {fold}",
        os.path.join(output_dir, "plots", f"final_task_score_t_dist_fold_{fold}.png"),
        xlabel="Transformer_t",
    )

    if len(domain_test) > 0:
        X_domain = transform_df(domain_test, scaler)
        domain_task_score = predict_score(best_model, X_domain)

        np.save(os.path.join(output_dir, "arrays", f"domain_task_score_fold_{fold}.npy"), domain_task_score)
        np.save(os.path.join(output_dir, "arrays", f"domain_label_fold_{fold}.npy"), domain_test["domain_label"].values)
        np.save(os.path.join(output_dir, "arrays", f"domain_weight_fold_{fold}.npy"), domain_test["weight"].values)

        sb_result = plot_sb_data_mc_ratio(
            domain_test,
            domain_task_score,
            os.path.join(output_dir, "plots", f"sb_data_mc_ratio_vs_final_score_fold_{fold}.png"),
            score_name="Transformer score",
            n_bins=10,
        )

        domain_task_score_t = tsf.transform(domain_task_score.reshape(-1, 1)).reshape(-1)
        np.save(os.path.join(output_dir, "arrays", f"domain_task_score_t_fold_{fold}.npy"), domain_task_score_t)

        sb_t_result = plot_sb_data_mc_ratio(
            domain_test,
            domain_task_score_t,
            os.path.join(output_dir, "plots", f"sb_data_mc_ratio_vs_final_score_t_fold_{fold}.png"),
            score_name="Transformer_t",
            n_bins=10,
        )

        sb_chi2 = float(sb_result["chi2_ndof"])
        sb_t_chi2 = float(sb_t_result["chi2_ndof"])
    else:
        sb_chi2 = np.nan
        sb_t_chi2 = np.nan

    fold_summaries.append({
        "fold": fold,
        "final_task_test_auc": float(final_test_auc),
        "sb_chi2_ndof": sb_chi2,
        "sb_t_chi2_ndof": sb_t_chi2,
        "n_train": int(len(task_train)),
        "n_val": int(len(task_val)),
        "n_test": int(len(task_test)),
        "n_domain_test": int(len(domain_test)),
    })

    with open(os.path.join(output_dir, "fold_summary.json"), "w") as f:
        json.dump(fold_summaries, f, indent=2)

    print(f"[INFO] Fold {fold}: finished and cleaned up", flush=True)

    del model
    del best_model
    tf.keras.backend.clear_session()
    gc.collect()


print("\n[DONE] FT-Transformer with mass no-domain GPU training finished.", flush=True)
print("[DONE] Output:", output_dir, flush=True)
print(json.dumps(fold_summaries, indent=2), flush=True)
