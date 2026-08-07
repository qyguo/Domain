#!/usr/bin/env python3

import os
import gc
import json
import pickle
import joblib
import argparse
from datetime import datetime

import uproot
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import roc_curve, auc, roc_auc_score


def parse_years(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() in ["none", "all"]:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train VBF Hmumu DNN models with optional domain adaptation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--save-mode", choices=["both", "domain", "nodomain"], default="both")
    parser.add_argument("--only-years", default="2022,2022EE,2023,2023BPix,2024,2025",
                        help="Comma-separated eras to train, e.g. 2024,2025. Use all for all eras.")
    parser.add_argument("--use-source-year", action="store_true",
                        help="Add numeric source_year as a trainable feature.")
    parser.add_argument("--no-era-features", action="store_true",
                        help="Do not add one-hot era_YYYY features.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Train only this fold. Useful for Condor array jobs.")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--prepare-folds-only", action="store_true",
                        help="Load ROOT files, build task/domain fold pickles, then exit.")
    parser.add_argument("--input-fold-dir", default=None,
                        help="Directory with task_df_with_folds.pkl and domain_df_with_folds.pkl.")
    parser.add_argument("--write-condor", action="store_true",
                        help="Write a Condor submit file that runs one job per fold, then exit.")
    parser.add_argument("--combine-fold-summaries", action="store_true",
                        help="Combine fold_summary_fold_N.json files into fold_summary.json, then exit.")
    parser.add_argument("--condor-cpus", type=int, default=4)
    parser.add_argument("--condor-gpus", type=int, default=0,
                        help="Request GPUs for each fold job. Keep 0 for CPU jobs.")
    parser.add_argument("--condor-memory", default="8GB")
    parser.add_argument("--condor-flavour", default="workday")
    parser.add_argument("--condor-workdir", default=None,
                        help="Working directory for Condor fold jobs. Defaults to current directory.")
    parser.add_argument(
        "--condor-env-setup",
        default="/cvmfs/sft.cern.ch/lcg/views/LCG_107_cuda/x86_64-el9-gcc11-opt/setup.sh",
        help="Environment setup script sourced inside each Condor job.",
    )
    parser.add_argument("--condor-require-el9", action="store_true", default=True,
                        help="Require an EL9 worker node for the LCG_107_cuda environment.")
    parser.add_argument("--no-condor-require-el9", dest="condor_require_el9", action="store_false")
    parser.add_argument("--output-base", default="/eos/user/q/qguo/SWAN_projects/ML_test/")
    parser.add_argument("--output-name", default=None)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--l2-reg", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clipnorm", type=float, default=5.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Enable mixed_float16. Best on GPUs; leave off for CPU-only jobs.")

    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--max-lambda", type=float, default=1.0)
    parser.add_argument("--focus-weight-max", type=float, default=3.0)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--domain-auc-penalty", type=float, default=0.02,
                        help="For DA model selection, subtract penalty*abs(domain_auc-0.5) from task AUC.")

    parser.add_argument("--skip-feature-importance", action="store_true")
    parser.add_argument("--feature-importance-repeats", type=int, default=1)
    parser.add_argument("--feature-importance-max-events", type=int, default=200000)

    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--inter-op-threads", type=int, default=2)
    return parser.parse_args()


args = parse_args()


# ============================================================
# Thread setup
# ============================================================

n_threads = args.n_threads
if n_threads is None:
    n_threads = int(os.environ.get("_CONDOR_NPROCS", os.environ.get("OMP_NUM_THREADS", "4")))
tf.config.threading.set_intra_op_parallelism_threads(n_threads)
tf.config.threading.set_inter_op_parallelism_threads(args.inter_op_threads)

if args.mixed_precision:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

print(f"TensorFlow intra_op threads = {n_threads}", flush=True)
print(f"TensorFlow inter_op threads = {args.inter_op_threads}", flush=True)
print(f"TensorFlow mixed precision = {args.mixed_precision}", flush=True)


# ============================================================
# Configuration
# ============================================================

# Save mode:
#   "both"     -> train and save both no-domain and domain models
#   "domain"   -> train and save only domain-adaptation models
#   "nodomain" -> train and save only no-domain models
SAVE_MODE = args.save_mode

if SAVE_MODE not in ["both", "domain", "nodomain"]:
    raise RuntimeError("SAVE_MODE must be one of: both, domain, nodomain")

TRAIN_NODOMAIN = SAVE_MODE in ["both", "nodomain"]
TRAIN_DOMAIN = SAVE_MODE in ["both", "domain"]
MODEL_VARIANT = "simple_dnn"

# One path per year / era
path = {}
path[0] = "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2022_ggHVBF/"
path[1] = "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2022EE_ggHVBF/"
path[2] = "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2023_ggHVBF/"
path[3] = "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2023BPix_ggHVBF/"
#path[4] = "/eos/user/z/zhangxu/sharing/hmm/2024_v4/skimmed_ntuples/SRSB_v2/"
#path[5] = "/eos/user/h/hakou/Hmumu_Share/qguo/2025/skimmed_ntuples/SRSB_noJetHornVeto/"
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

signal_files_by_path = {
    0: {"VBFHToMuMu_M125_ggHUnc.root": "SIGNAL"},
    1: {"VBFHToMuMu_M125_ggHUnc.root": "SIGNAL"},
    2: {"VBFHToMuMu_M125_ggHUnc.root": "SIGNAL"},
    3: {"VBFHToMuMu_M125_ggHUnc.root": "SIGNAL"},
    4: {"VBFHToMuMu_M125.root": "SIGNAL"},
    5: {"VBFHToMuMu_M125.root": "SIGNAL"},
}

background_files_by_path = {
    0: {
        #"DY_105To160.root": "DY",
        "DY_105To160_Inc_failvbffilter.root": "DY",
        "DY_105To160_Fil-VBF_passvbffilter.root": "DY",
        "EWK_LLJJ_M105To160.root": "EWK_ZJJ",
        "TTTo2L2Nu.root": "TT",
    },
    1: {
        #"DY_105To160.root": "DY",
        "DY_105To160_Inc_failvbffilter.root": "DY",
        "DY_105To160_Fil-VBF_passvbffilter.root": "DY",
        "EWK_LLJJ_M105To160.root": "EWK_ZJJ",
        "TTTo2L2Nu.root": "TT",
    },
    2: {
        #"DY_105To160.root": "DY",
        "DY_105To160_Inc_failvbffilter.root": "DY",
        "DY_105To160_Fil-VBF_passvbffilter.root": "DY",
        "EWK_LLJJ_M105To160.root": "EWK_ZJJ",
        "TTTo2L2Nu.root": "TT",
    },
    3: {
        #"DY_105To160.root": "DY",
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
    0: {"data.root": "DATA"},
    1: {"data.root": "DATA"},
    2: {"data.root": "DATA"},
    3: {"data.root": "DATA"},
    4: {"data.root": "DATA"},
    5: {"data.root": "DATA"},
}

## Quick switch: train one year only
#ONLY_YEAR = "2024"
## ONLY_YEAR = None
#
#if ONLY_YEAR is not None:
#    keep_keys = [k for k, v in era_names.items() if v == ONLY_YEAR]
#    path = {k: v for k, v in path.items() if k in keep_keys}
#    era_names = {k: v for k, v in era_names.items() if k in keep_keys}
#    tree_name_by_path = {k: v for k, v in tree_name_by_path.items() if k in keep_keys}
#    signal_files_by_path = {k: v for k, v in signal_files_by_path.items() if k in keep_keys}
#    background_files_by_path = {k: v for k, v in background_files_by_path.items() if k in keep_keys}
#    data_files_by_path = {k: v for k, v in data_files_by_path.items() if k in keep_keys}

ONLY_YEARS = parse_years(args.only_years)

if ONLY_YEARS is not None:
    keep_keys = [k for k, v in era_names.items() if v in ONLY_YEARS]
    if len(keep_keys) == 0:
        raise RuntimeError(f"No configured era matched --only-years={args.only_years}")

    path = {k: v for k, v in path.items() if k in keep_keys}
    era_names = {k: v for k, v in era_names.items() if k in keep_keys}
    tree_name_by_path = {k: v for k, v in tree_name_by_path.items() if k in keep_keys}
    signal_files_by_path = {k: v for k, v in signal_files_by_path.items() if k in keep_keys}
    background_files_by_path = {k: v for k, v in background_files_by_path.items() if k in keep_keys}
    data_files_by_path = {k: v for k, v in data_files_by_path.items() if k in keep_keys}


# ============================================================
# Branches / features
# ============================================================

target_cate_index = 3

mass_branch = "diMufsr_kit_BSC_mass"
weight_branch = "eventWeight"

sr_low = 115.0
sr_high = 135.0

FIX_DOMAIN_MASS_TO_125 = True
DOMAIN_FIXED_MASS_VALUE = 125.0

EVENT_ID_CANDIDATES = [
    ["run", "lumi", "event"],
]

# Keep one canonical feature naming convention for the model. 2022-2023
# ntuples store the dimuon BSC variables with rc names, while 2024-2025 use
# kit names. The loader below maps whichever branch exists into these
# canonical 2024/2025-style column names.
BRANCH_ALIASES = {
    "diMufsr_kit_BSC_mass": ["diMufsr_kit_BSC_mass", "diMufsr_rc_BSC_mass"],
    "diMufsr_kit_BSC_pt": ["diMufsr_kit_BSC_pt", "diMufsr_rc_BSC_pt"],
    "diMufsr_kit_BSC_eta": ["diMufsr_kit_BSC_eta", "diMufsr_rc_BSC_eta"],
    "log_diMufsr_kit_BSC_pt": ["log_diMufsr_kit_BSC_pt", "log_diMufsr_rc_BSC_pt"],
}

# These branches are useful to keep in the cached fold data when available,
# but their availability is not uniform:
#   2022-2023: genvbffilter_flag/n_jets_matched_genjet exist only for DY.
#   2024-2025: they exist for MC, not for data.
OPTIONAL_BRANCH_DEFAULTS = {
    "source_year": np.nan,
    "genvbffilter_flag": -1,
    "n_jets_matched_genjet": -1,
}

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

n_folds = args.n_folds
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
dropout = args.dropout
l2_reg = args.l2_reg
weight_decay = args.weight_decay
clipnorm = args.clipnorm

# Domain adaptation parameters
alpha = args.alpha
max_lambda = args.max_lambda
focus_weight_max = args.focus_weight_max
warmup_epochs = args.warmup_epochs
domain_auc_penalty = args.domain_auc_penalty


# ============================================================
# Output
# ============================================================

current_date = datetime.now().strftime("%m%d")
default_output_name = f"noBr4_{MODEL_VARIANT}_DAandNoDomain_2026{current_date}_signedPhysW_{SAVE_MODE}_v2"
output_name = args.output_name or default_output_name
path_out = args.output_base
output_dir = os.path.abspath(os.path.join(path_out, output_name))

os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/models", exist_ok=True)
os.makedirs(f"{output_dir}/plots", exist_ok=True)
os.makedirs(f"{output_dir}/arrays", exist_ok=True)
os.makedirs(f"{output_dir}/condor", exist_ok=True)

print("output_dir =", output_dir, flush=True)
print("SAVE_MODE =", SAVE_MODE, flush=True)
print("MODEL_VARIANT =", MODEL_VARIANT, flush=True)
print("ONLY_YEARS =", ONLY_YEARS, flush=True)
print("use_source_year =", args.use_source_year, flush=True)


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


def era_to_source_year(era):
    text = str(era)
    if text.startswith("2022"):
        return 2022.0
    if text.startswith("2023"):
        return 2023.0
    if text.startswith("2024"):
        return 2024.0
    if text.startswith("2025"):
        return 2025.0
    return np.nan


def ensure_source_year(df_in):
    out = df_in.copy()
    derived = out["era"].map(era_to_source_year).astype(float)
    if "source_year" not in out.columns:
        out["source_year"] = derived
    else:
        out["source_year"] = pd.to_numeric(out["source_year"], errors="coerce")
        out["source_year"] = out["source_year"].fillna(derived)
    return out


def build_year_features(df_in):
    features = []
    if not args.no_era_features:
        for era in era_names.values():
            col = f"era_{era}"
            df_in[col] = (df_in["era"] == era).astype(float)
            features.append(col)
    if args.use_source_year:
        df_in["source_year"] = pd.to_numeric(df_in["source_year"], errors="coerce")
        features.append("source_year")
    return features


def is_run2_or_2023_era(era):
    text = str(era)
    return text.startswith("2022") or text.startswith("2023")


def branch_candidates(branch, era):
    candidates = list(BRANCH_ALIASES.get(branch, [branch]))

    if is_run2_or_2023_era(era) and branch in BRANCH_ALIASES:
        rc_first = [c for c in candidates if "_rc_" in c]
        others = [c for c in candidates if c not in rc_first]
        candidates = rc_first + others

    out = []
    for cand in candidates:
        if cand not in out:
            out.append(cand)
    return out


def resolve_branches_for_tree(requested_branches, available, era):
    read_branches = []
    branch_map = {}
    missing = []

    for branch in requested_branches:
        actual = None
        for cand in branch_candidates(branch, era):
            if cand in available:
                actual = cand
                break

        if actual is None:
            missing.append(branch)
            continue

        branch_map[branch] = actual
        read_branches.append(actual)

    return sorted(set(read_branches)), branch_map, missing


def apply_branch_aliases(df_in, requested_branches, branch_map, rec):
    out = df_in.copy()
    alias_columns_to_drop = []

    for canonical in requested_branches:
        actual = branch_map.get(canonical)
        if actual is None:
            continue

        if actual != canonical:
            out[canonical] = out[actual]
            alias_columns_to_drop.append(actual)

    for branch, default in OPTIONAL_BRANCH_DEFAULTS.items():
        if branch not in out.columns:
            out[branch] = default

    aliases_used = [
        (canonical, actual)
        for canonical, actual in branch_map.items()
        if canonical != actual
    ]
    if aliases_used:
        print("[INFO] branch aliases used for", rec["path"], flush=True)
        for canonical, actual in aliases_used:
            print(f"       {canonical} <- {actual}", flush=True)

    for col in alias_columns_to_drop:
        if col in out.columns and col not in requested_branches:
            out = out.drop(columns=[col])

    return out


def write_condor_files():
    script_path = os.path.abspath(__file__)
    cwd = args.condor_workdir or os.getcwd()
    condor_dir = os.path.join(output_dir, "condor")
    os.makedirs(condor_dir, exist_ok=True)

    run_sh = os.path.join(condor_dir, "run_fold.sh")
    submit_file = os.path.join(condor_dir, "submit_folds.sub")
    folds = " ".join(str(i) for i in range(1, n_folds + 1))

    source_year_flag = " --use-source-year" if args.use_source_year else ""
    era_flag = " --no-era-features" if args.no_era_features else ""
    skip_fi_flag = " --skip-feature-importance" if args.skip_feature_importance else ""
    mixed_precision_flag = " --mixed-precision" if args.mixed_precision else ""

    command = (
        f'python3 "{script_path}" '
        f'--input-fold-dir "{output_dir}" '
        f'--output-base "{path_out}" '
        f'--output-name "{output_name}" '
        f'--save-mode "{SAVE_MODE}" '
        f'--only-years "{args.only_years}" '
        f'--n-folds {n_folds} '
        f'--fold "${{FOLD}}" '
        f'--epochs {epochs} '
        f'--batch-size {batch_size} '
        f'--learning-rate {learning_rate} '
        f'--dropout {dropout} '
        f'--l2-reg {l2_reg} '
        f'--weight-decay {weight_decay} '
        f'--clipnorm {clipnorm} '
        f'--early-stopping-patience {args.early_stopping_patience} '
        f'--min-delta {args.min_delta} '
        f'--alpha {alpha} '
        f'--max-lambda {max_lambda} '
        f'--focus-weight-max {focus_weight_max} '
        f'--warmup-epochs {warmup_epochs} '
        f'--domain-auc-penalty {domain_auc_penalty} '
        f'--feature-importance-repeats {args.feature_importance_repeats} '
        f'--feature-importance-max-events {args.feature_importance_max_events} '
        f'--n-threads {args.condor_cpus} '
        f'--inter-op-threads {args.inter_op_threads}'
        f'{source_year_flag}{era_flag}{skip_fi_flag}{mixed_precision_flag}'
    )

    with open(run_sh, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -euo pipefail\n")
        if args.condor_env_setup:
            f.write(f'source "{args.condor_env_setup}"\n')
        f.write(f"export OMP_NUM_THREADS={args.condor_cpus}\n")
        f.write(f"export MKL_NUM_THREADS={args.condor_cpus}\n")
        f.write("export TF_CPP_MIN_LOG_LEVEL=1\n")
        f.write("FOLD=\"$1\"\n")
        f.write(f'cd "{cwd}"\n')
        f.write("python3 --version\n")
        f.write("python3 - <<'PYENV'\n")
        f.write("import tensorflow as tf\n")
        f.write("print('TensorFlow', tf.__version__)\n")
        f.write("print('GPUs', tf.config.list_physical_devices('GPU'))\n")
        f.write("PYENV\n")
        f.write(f"{command}\n")
    os.chmod(run_sh, 0o755)

    with open(submit_file, "w") as f:
        f.write(f"executable = {run_sh}\n")
        f.write("arguments = $(Fold)\n")
        f.write(f"request_cpus = {args.condor_cpus}\n")
        if args.condor_gpus > 0:
            f.write(f"request_gpus = {args.condor_gpus}\n")
        f.write(f"request_memory = {args.condor_memory}\n")
        f.write(f'+JobFlavour = "{args.condor_flavour}"\n')
        if args.condor_require_el9:
            f.write('requirements = (OpSysAndVer =?= "AlmaLinux9")\n')
        f.write("getenv = True\n")
        f.write(f"output = {condor_dir}/fold_$(Fold).out\n")
        f.write(f"error = {condor_dir}/fold_$(Fold).err\n")
        f.write(f"log = {condor_dir}/fold_$(Fold).log\n")
        f.write(f"queue Fold in {folds}\n")

    print("[CONDOR] wrote:", run_sh, flush=True)
    print("[CONDOR] wrote:", submit_file, flush=True)
    print("[CONDOR] first run with --prepare-folds-only, then submit:", flush=True)
    print(f"condor_submit {submit_file}", flush=True)


def combine_fold_summaries():
    summaries = []
    missing = []
    for fold in range(1, n_folds + 1):
        fold_summary_path = os.path.join(output_dir, f"fold_summary_fold_{fold}.json")
        if not os.path.exists(fold_summary_path):
            missing.append(fold_summary_path)
            continue
        with open(fold_summary_path) as f:
            summaries.extend(json.load(f))

    outpath = os.path.join(output_dir, "fold_summary.json")
    with open(outpath, "w") as f:
        json.dump(to_jsonable(summaries), f, indent=2)

    print("[INFO] wrote combined summary:", outpath, flush=True)
    if missing:
        print("[WARNING] missing fold summary files:", flush=True)
        for path_missing in missing:
            print("  ", path_missing, flush=True)


if args.write_condor:
    write_condor_files()
    raise SystemExit(0)

if args.combine_fold_summaries:
    combine_fold_summaries()
    raise SystemExit(0)


# ============================================================
# ROOT loading
# ============================================================

def build_file_records():
    records = []

    for ipath, folder in path.items():
        era = era_names[ipath]
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
            read_branches, branch_map, missing = resolve_branches_for_tree(
                branches,
                available,
                rec["era"],
            )

            missing_required = [b for b in missing if b not in OPTIONAL_BRANCH_DEFAULTS]
            missing_optional = [b for b in missing if b in OPTIONAL_BRANCH_DEFAULTS]
            if len(missing_required) > 0:
                print("[WARNING] missing branches in", fpath, flush=True)
                print(missing_required, flush=True)
            if len(missing_optional) > 0:
                print("[INFO] optional branches missing in", fpath, flush=True)
                print(missing_optional, flush=True)

            df_tmp = tree.arrays(read_branches, library="pd")
            df_tmp = apply_branch_aliases(df_tmp, branches, branch_map, rec)

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


def load_or_build_datasets():
    global all_feature_branches, year_features

    if args.input_fold_dir is not None:
        input_dir = os.path.abspath(args.input_fold_dir)
        features_path = os.path.join(input_dir, "features.json")
        if not os.path.exists(features_path):
            raise RuntimeError(f"Missing cached feature config: {features_path}")
        with open(features_path) as f:
            feature_config = json.load(f)
        year_features = feature_config["year_features"]
        all_feature_branches = feature_config["all_feature_branches"]

        task_path = os.path.join(input_dir, "task_df_with_folds.pkl")
        domain_path = os.path.join(input_dir, "domain_df_with_folds.pkl")
        task_cached = pd.read_pickle(task_path)
        domain_cached = pd.read_pickle(domain_path)
        print("[INFO] loaded cached task folds:", task_path, len(task_cached), flush=True)
        print("[INFO] loaded cached domain folds:", domain_path, len(domain_cached), flush=True)
        return task_cached, domain_cached

    records = build_file_records()
    df = load_data_from_records(records, branches)

    print("Loaded events:", len(df), flush=True)

    if "cate_index" in df.columns:
        before = len(df)
        df = df[df["cate_index"] == target_cate_index].copy()
        print(f"After cate_index == {target_cate_index}: {before} -> {len(df)}", flush=True)

    if weight_branch not in df.columns:
        raise RuntimeError(f"Missing weight branch: {weight_branch}")

    df = ensure_source_year(df)
    df["signed_weight"] = df[weight_branch].astype(float)
    df.loc[df["sample_type"] == "data", "signed_weight"] = 1.0
    df["signed_weight"] = df["signed_weight"].replace([np.inf, -np.inf], np.nan)
    df["signed_weight"] = df["signed_weight"].fillna(0.0)
    df["weight"] = df["signed_weight"]

    df["is_sr"] = ((df[mass_branch] > sr_low) & (df[mass_branch] < sr_high)).astype(int)
    df["is_sb"] = ((df[mass_branch] < sr_low) | (df[mass_branch] > sr_high)).astype(int)

    year_features = build_year_features(df)
    all_feature_branches = feature_branches + year_features

    for col in all_feature_branches:
        if col not in df.columns:
            raise RuntimeError(f"Missing feature branch: {col}")

    df[all_feature_branches] = df[all_feature_branches].replace([np.inf, -np.inf], np.nan)
    df[all_feature_branches] = df[all_feature_branches].fillna(-1.0)

    with open(f"{output_dir}/features.json", "w") as f:
        json.dump(to_jsonable({
            "feature_branches": feature_branches,
            "year_features": year_features,
            "all_feature_branches": all_feature_branches,
            "mass_branch": mass_branch,
            "weight_branch": weight_branch,
            "sr_low": sr_low,
            "sr_high": sr_high,
            "target_cate_index": target_cate_index,
            "branch_aliases": BRANCH_ALIASES,
            "MODEL_VARIANT": MODEL_VARIANT,
            "model": "single shared DNN classifier with optional GRL domain head",
            "domain": "data sideband vs background-MC sideband",
            "FIX_DOMAIN_MASS_TO_125": FIX_DOMAIN_MASS_TO_125,
            "DOMAIN_FIXED_MASS_VALUE": DOMAIN_FIXED_MASS_VALUE,
            "use_source_year": args.use_source_year,
            "no_era_features": args.no_era_features,
            "ONLY_YEARS": ONLY_YEARS,
        }), f, indent=2)

    task = df[
        (df["is_sr"] == 1)
        & (df["sample_type"].isin(["signal", "background"]))
    ].copy()
    task["task_label"] = (task["sample_type"] == "signal").astype(float)

    domain = df[
        (df["is_sb"] == 1)
        & (df["sample_type"].isin(["data", "background"]))
    ].copy()
    domain["domain_label"] = (domain["sample_type"] == "data").astype(float)

    if len(task) == 0:
        raise RuntimeError("task_df is empty. Check signal/background and SR selection.")
    if TRAIN_DOMAIN and len(domain) == 0:
        raise RuntimeError("domain_df is empty. Add data files and check SB selection.")

    print("\n[Dataset sizes]", flush=True)
    print("Task signal vs all background:", len(task), flush=True)
    print(task.groupby(["sample_type", "process"])["signed_weight"].sum(), flush=True)
    print("\nDomain data vs MC in SB:", len(domain), flush=True)
    print(domain.groupby(["sample_type", "process"])["signed_weight"].sum(), flush=True)

    task = add_signed_train_weight(task, "task_label")
    domain = add_positive_train_weight(domain, "domain_label")
    task = assign_folds(task, "task_label", n_folds=n_folds, random_state=33)
    domain = assign_folds(domain, "domain_label", n_folds=n_folds, random_state=44)

    task.to_pickle(f"{output_dir}/task_df_with_folds.pkl")
    domain.to_pickle(f"{output_dir}/domain_df_with_folds.pkl")
    return task, domain


# ============================================================
# Training weights
# ============================================================

def add_signed_train_weight(df_in, label_col):
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


def add_positive_train_weight(df_in, label_col):
    out = df_in.copy()
    out["train_weight"] = np.abs(out["signed_weight"].astype(float))

    labels = sorted(out[label_col].unique())
    n_class = len(labels)

    for label in labels:
        mask = out[label_col] == label
        sum_w = np.sum(out.loc[mask, "train_weight"].values)
        if sum_w > 0:
            out.loc[mask, "train_weight"] *= len(out) / (n_class * sum_w)

    out["train_weight"] = out["train_weight"].replace([np.inf, -np.inf], np.nan)
    out["train_weight"] = out["train_weight"].fillna(0.0)
    return out


# ============================================================
# Folds
# ============================================================

def assign_folds(df_in, label_col, n_folds=4, random_state=42):
    out = df_in.copy()
    out["fold_id"] = -1

    counts = out[label_col].value_counts()
    if len(counts) < 2:
        raise RuntimeError(f"{label_col} has fewer than two classes.")
    if counts.min() < n_folds:
        raise RuntimeError(f"{label_col} has a class with fewer than {n_folds} events.")

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    indices = np.arange(len(out))

    for fold, (_, test_idx) in enumerate(splitter.split(indices, out[label_col].values), start=1):
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


task_df, domain_df = load_or_build_datasets()

print("[INFO] task event id columns:", get_event_id_cols(task_df), flush=True)
print("[INFO] domain event id columns:", get_event_id_cols(domain_df), flush=True)

if args.prepare_folds_only:
    print("[INFO] prepared fold cache only; exiting before training.", flush=True)
    print("Output:", output_dir, flush=True)
    raise SystemExit(0)


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


def lambda_schedule(epoch, max_epochs, max_lambda=1.0, warmup_epochs=0):
    if epoch <= warmup_epochs:
        return 0.0

    active_epochs = max(max_epochs - warmup_epochs, 1)
    p = (epoch - warmup_epochs) / float(active_epochs)
    p = np.clip(p, 0.0, 1.0)
    return max_lambda * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


# ============================================================
# Simple DNN model
# ============================================================

def dense_layer(units, name):
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    return tf.keras.layers.Dense(
        units,
        activation="relu",
        kernel_regularizer=regularizer,
        name=name,
    )


def build_simple_dnn_model(n_features, dropout=0.2, use_domain=True):
    inputs = tf.keras.layers.Input(shape=(n_features,), name="features")

    shared = dense_layer(128, "shared_dense1")(inputs)
    shared = tf.keras.layers.Dropout(dropout, name="shared_dropout1")(shared)
    shared = dense_layer(64, "shared_dense2")(shared)
    shared = tf.keras.layers.Dropout(dropout, name="shared_dropout2")(shared)
    shared = dense_layer(32, "shared_dense3")(shared)
    shared = tf.keras.layers.LayerNormalization(name="shared_layernorm")(shared)

    task = dense_layer(16, "task_dense")(shared)
    task_output = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="Task_Output")(task)

    outputs = [task_output]
    grl = None

    if use_domain:
        grl = GradientReversalLayer(name="Gradient_Reversal")
        reversed_shared = grl(shared)
        domain = dense_layer(16, "domain_dense")(reversed_shared)
        domain = tf.keras.layers.Dropout(dropout, name="domain_dropout")(domain)
        domain_output = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="Domain_Output")(domain)
        outputs.append(domain_output)

    model_name = "Simple_DNN_with_Domain_Adaptation" if use_domain else "Simple_DNN_noDomain"
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name), grl


def build_optimizer():
    kwargs = {}
    if clipnorm and clipnorm > 0:
        kwargs["clipnorm"] = clipnorm
    if weight_decay and weight_decay > 0 and hasattr(tf.keras.optimizers, "AdamW"):
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )
    return tf.keras.optimizers.Adam(learning_rate=learning_rate, **kwargs)


# ============================================================
# Array helpers and loss
# ============================================================

def fit_scaler_from_task_train(task_train):
    scaler = StandardScaler()
    scaler.fit(task_train[all_feature_branches].values)
    return scaler


def copy_fix_domain_mass(df_in):
    out = df_in.copy()
    if FIX_DOMAIN_MASS_TO_125:
        out[mass_branch] = DOMAIN_FIXED_MASS_VALUE
    return out


def transform_df(df_in, scaler, fix_domain_mass=False):
    df_work = copy_fix_domain_mass(df_in) if fix_domain_mass else df_in
    X_scaled = scaler.transform(df_work[all_feature_branches].values)
    return X_scaled.astype(np.float32)


def make_batches(X, y, w, batch_size, shuffle=True):
    n = len(y)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)

    for start in range(0, n, batch_size):
        sub = idx[start:start + batch_size]
        yield (
            X[sub].astype(np.float32),
            y[sub].astype(np.float32).reshape(-1, 1),
            w[sub].astype(np.float32).reshape(-1, 1),
        )


def bce_loss_signed_weight(y_true, y_pred, weight):
    loss = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    loss = tf.reshape(loss, (-1, 1))
    numerator = tf.reduce_sum(loss * weight)
    denominator = tf.reduce_sum(tf.abs(weight)) + 1e-8
    return numerator / denominator


# ============================================================
# Training / prediction
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
    alpha=2.0,
    max_lambda=0.7,
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

    n_task = int(np.ceil(len(arrays["task"][1]) / batch_size))

    if use_domain:
        n_domain = int(np.ceil(len(arrays["domain"][1]) / batch_size))
        n_steps = min(n_task, n_domain)
    else:
        n_steps = n_task

    if n_steps <= 0:
        raise RuntimeError("n_steps <= 0. One of the training datasets is empty.")

    task_iter = make_batches(*arrays["task"], batch_size=batch_size, shuffle=True)
    domain_iter = make_batches(*arrays["domain"], batch_size=batch_size, shuffle=True) if use_domain else None

    losses = {"total": [], "task": [], "domain": []}

    for i in range(n_steps):
        tk_x, tk_y, tk_w = next(task_iter)

        if use_domain:
            dm_x, dm_y, dm_w = next(domain_iter)

        with tf.GradientTape() as tape:
            task_outputs = model({"features": tk_x}, training=True)
            task_pred = task_outputs[0] if isinstance(task_outputs, list) else task_outputs
            loss_task = bce_loss_signed_weight(tk_y, task_pred, tk_w)

            if use_domain:
                domain_outputs = model({"features": dm_x}, training=True)
                dm_task_score = domain_outputs[0]
                dm_domain_pred = domain_outputs[1]

                focus_weight = 1.0 + alpha * tf.stop_gradient(tf.clip_by_value(dm_task_score, 0.0, 1.0))
                focus_weight = tf.clip_by_value(focus_weight, 1.0, focus_weight_max)
                loss_domain = bce_loss_signed_weight(dm_y, dm_domain_pred, dm_w * focus_weight)
                loss_domain /= 2.0 
                loss_total = loss_task + grl_lambda * loss_domain
            else:
                loss_domain = tf.constant(0.0, dtype=tf.float32)
                loss_total = loss_task

            if model.losses:
                loss_total = loss_total + tf.add_n(model.losses)

        grads = tape.gradient(loss_total, model.trainable_variables)
        grads_and_vars = [
            (g, v) for g, v in zip(grads, model.trainable_variables)
            if g is not None
        ]
        optimizer.apply_gradients(grads_and_vars)

        losses["total"].append(float(loss_total.numpy()))
        losses["task"].append(float(loss_task.numpy()))
        losses["domain"].append(float(loss_domain.numpy()))

        if (i + 1) % 100 == 0:
            print(f"    epoch {epoch:03d}: finished step {i + 1}/{n_steps}", flush=True)

    out = {k: float(np.mean(v)) for k, v in losses.items()}
    out["lambda"] = float(grl_lambda)
    out["n_steps"] = int(n_steps)
    return out


def predict_outputs(model, X, use_domain=True, batch_size=4096):
    n_out = 2 if use_domain else 1
    outs = [[] for _ in range(n_out)]

    for start in range(0, len(X), batch_size):
        xb = X[start:start + batch_size].astype(np.float32)
        pred = model.predict({"features": xb}, verbose=0, batch_size=batch_size)
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


# ============================================================
# Plots
# ============================================================

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


def subsample_for_importance(X, df_eval, max_events, random_state):
    if max_events is None or max_events <= 0 or len(df_eval) <= max_events:
        return X, df_eval
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(df_eval), size=max_events, replace=False)
    return X[idx], df_eval.iloc[idx].copy()


def permutation_feature_importance(
    model,
    X,
    df_eval,
    feature_names,
    y_col,
    w_col,
    use_domain,
    out_prefix,
    n_repeats=1,
    max_events=200000,
    random_state=12345,
):
    if n_repeats <= 0:
        return None

    X_eval, df_eval_small = subsample_for_importance(X, df_eval, max_events, random_state)
    y = df_eval_small[y_col].values
    w = df_eval_small[w_col].values
    baseline_score = predict_outputs(model, X_eval, use_domain=use_domain)[0]
    baseline_auc = safe_auc(y, baseline_score, w)

    rng = np.random.default_rng(random_state)
    rows = []
    for i, feature in enumerate(feature_names):
        auc_values = []
        for _ in range(n_repeats):
            X_perm = X_eval.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            perm_score = predict_outputs(model, X_perm, use_domain=use_domain)[0]
            auc_values.append(safe_auc(y, perm_score, w))
        mean_auc = float(np.nanmean(auc_values))
        std_auc = float(np.nanstd(auc_values))
        rows.append({
            "feature": feature,
            "baseline_auc": float(baseline_auc),
            "permuted_auc_mean": mean_auc,
            "permuted_auc_std": std_auc,
            "importance_auc_drop": float(baseline_auc - mean_auc),
        })

    imp = pd.DataFrame(rows).sort_values("importance_auc_drop", ascending=False)
    csv_path = f"{out_prefix}.csv"
    png_path = f"{out_prefix}.png"
    imp.to_csv(csv_path, index=False)

    plot_df = imp.head(min(20, len(imp))).iloc[::-1]
    plt.figure(figsize=(9, max(5, 0.35 * len(plot_df) + 1.5)))
    plt.barh(plot_df["feature"], plot_df["importance_auc_drop"])
    plt.xlabel("AUC drop after permutation")
    plt.title("Permutation feature importance")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    print("[INFO] feature importance:", csv_path, flush=True)
    print("[INFO] feature importance:", png_path, flush=True)
    return imp


def plot_training_history(history, fold, tag):
    epochs_arr = np.arange(1, len(history["loss_total"]) + 1)

    plt.figure(figsize=(9, 6))
    plt.plot(epochs_arr, history["loss_total"], label="total")
    plt.plot(epochs_arr, history["loss_task"], label="task")
    if "loss_domain" in history:
        plt.plot(epochs_arr, history["loss_domain"], label="domain")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training losses, {tag}, fold {fold}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/losses_{tag}_fold_{fold}.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_arr, history["auc_task_val"], label="task val AUC")
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
    ratio = []
    ratio_err = []
    data_y = []
    mc_y = []

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
    ax1.set_title("SB Data/MC vs simple DNN score")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    good = np.isfinite(ratio) & np.isfinite(ratio_err) & (ratio_err >= 0)
    ax2.errorbar(centers[good], ratio[good], yerr=ratio_err[good], fmt="o")
    ax2.axhline(1.0, linestyle="--", color="black")
    ax2.set_xlabel("Simple DNN score")
    ax2.set_ylabel("Data / MC")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ============================================================
# Train one variant
# ============================================================

def train_variant_for_fold(
    fold,
    tag,
    use_domain,
    arrays_train,
    arrays_val,
    arrays_test,
    dfs_val_test,
):
    print("\n" + "-" * 90, flush=True)
    print(f"Start variant: {tag}, fold {fold}, use_domain={use_domain}", flush=True)
    print("-" * 90, flush=True)

    model, grl = build_simple_dnn_model(
        n_features=len(all_feature_branches),
        dropout=dropout,
        use_domain=use_domain,
    )
    optimizer = build_optimizer()
    model.summary(print_fn=lambda x: print(x, flush=True))

    history = {
        "loss_total": [],
        "loss_task": [],
        "loss_domain": [],
        "lambda": [],
        "auc_task_val": [],
        "selection_metric": [],
    }
    if use_domain:
        history["auc_domain_val"] = []

    best_auc = -999.0
    best_metric = -999.0
    best_epoch = 0
    wait_epochs = 0
    if use_domain:
        best_model_path = f"{output_dir}/models/simple_DA_model_fold_{fold}.keras"
    else:
        best_model_path = f"{output_dir}/models/simple_noDomain_model_fold_{fold}.keras"

    for epoch in range(1, epochs + 1):
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

        task_val_x = arrays_val["task_X"]
        task_val = dfs_val_test["task_val"]
        val_outputs = predict_outputs(model, task_val_x, use_domain=use_domain)
        val_task_score = val_outputs[0]

        auc_task_val = safe_auc(
            task_val["task_label"].values,
            val_task_score,
            task_val["train_weight"].values,
        )

        if use_domain:
            domain_val_x = arrays_val["domain_X"]
            domain_val = dfs_val_test["domain_val"]
            val_domain_outputs = predict_outputs(model, domain_val_x, use_domain=True)
            val_domain_score = val_domain_outputs[1]
            auc_domain_val = safe_auc(
                domain_val["domain_label"].values,
                val_domain_score,
                domain_val["train_weight"].values,
            )
        else:
            auc_domain_val = np.nan

        selection_metric = auc_task_val
        if use_domain and np.isfinite(auc_domain_val):
            selection_metric = auc_task_val - domain_auc_penalty * abs(auc_domain_val - 0.5)
        if not np.isfinite(selection_metric):
            selection_metric = -np.inf

        history["loss_total"].append(metrics["total"])
        history["loss_task"].append(metrics["task"])
        history["loss_domain"].append(metrics["domain"])
        history["lambda"].append(metrics["lambda"])
        history["auc_task_val"].append(auc_task_val)
        history["selection_metric"].append(selection_metric)
        if use_domain:
            history["auc_domain_val"].append(auc_domain_val)

        print(
            f"{tag} Fold {fold} | Epoch {epoch:03d} | "
            f"steps={metrics['n_steps']} | "
            f"loss={metrics['total']:.5f} | "
            f"task={metrics['task']:.5f} | "
            f"domain={metrics['domain']:.5f} | "
            f"lambda={metrics['lambda']:.3f} | "
            f"val_task_auc={auc_task_val:.5f} | "
            f"val_domain_auc={auc_domain_val:.5f} | "
            f"select={selection_metric:.5f}",
            flush=True,
        )

        if selection_metric > best_metric + args.min_delta:
            best_metric = selection_metric
            best_auc = auc_task_val
            best_epoch = epoch
            wait_epochs = 0
            model.save(best_model_path)
            print(
                f"{tag} fold {fold}: saved best model "
                f"(epoch {best_epoch}, val AUC {best_auc:.5f}, select {best_metric:.5f})",
                flush=True,
            )
        else:
            wait_epochs += 1
            if args.early_stopping_patience > 0 and wait_epochs >= args.early_stopping_patience:
                print(
                    f"{tag} fold {fold}: early stopping at epoch {epoch}; "
                    f"best epoch {best_epoch}",
                    flush=True,
                )
                break

    if not os.path.exists(best_model_path):
        raise RuntimeError(f"No best model was saved for {tag} fold {fold}. Check validation AUC.")

    with open(f"{output_dir}/models/history_{tag}_fold_{fold}.pkl", "wb") as f:
        pickle.dump(history, f)

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

    task_test_x = arrays_test["task_X"]
    task_test = dfs_val_test["task_test"]
    test_outputs = predict_outputs(best_model, task_test_x, use_domain=use_domain)
    final_score = test_outputs[0]

    y_task_test = task_test["task_label"].values
    w_task_test_train = task_test["train_weight"].values
    w_task_test_phys = task_test["signed_weight"].values
    final_test_auc = safe_auc(y_task_test, final_score, w_task_test_train)

    if use_domain:
        domain_test_x = arrays_test["domain_X"]
        domain_test = dfs_val_test["domain_test"]
        domain_outputs = predict_outputs(best_model, domain_test_x, use_domain=True)
        domain_task_score = domain_outputs[0]
        domain_score = domain_outputs[1]

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

    print(f"[{tag} fold {fold}] simple DNN task test AUC = {final_test_auc:.6f}", flush=True)
    print(f"[{tag} fold {fold}] domain test AUC          = {domain_test_auc:.6f}", flush=True)

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
        f"{tag} simple DNN task ROC, fold {fold}",
        f"{output_dir}/plots/{tag}_simple_task_roc_fold_{fold}.png",
    )

    plot_score_distribution(
        y_task_test,
        final_score,
        w_task_test_phys,
        f"{tag} simple DNN score distribution, fold {fold}",
        f"{output_dir}/plots/{tag}_simple_task_score_dist_fold_{fold}.png",
        names=("background MC", "signal MC"),
    )

    plot_score_vs_mass(
        task_test,
        final_score,
        f"{tag} simple DNN score vs m_mumu, fold {fold}",
        f"{output_dir}/plots/{tag}_simple_score_vs_mass_fold_{fold}.png",
    )

    feature_importance_path = None
    if not args.skip_feature_importance:
        out_prefix = f"{output_dir}/plots/{tag}_feature_importance_fold_{fold}"
        permutation_feature_importance(
            best_model,
            task_test_x,
            task_test,
            all_feature_branches,
            y_col="task_label",
            w_col="train_weight",
            use_domain=use_domain,
            out_prefix=out_prefix,
            n_repeats=args.feature_importance_repeats,
            max_events=args.feature_importance_max_events,
            random_state=12345 + fold,
        )
        feature_importance_path = f"{out_prefix}.png"

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
            f"{output_dir}/plots/{tag}_sb_data_mc_ratio_vs_simple_score_fold_{fold}.png",
            n_bins=10,
        )

    tsf = QuantileTransformer(
        n_quantiles=min(1000, len(final_score)),
        output_distribution="uniform",
        subsample=1000000000,
        random_state=0,
    )
    tsf.fit(final_score.reshape(-1, 1))

    tsf_name = f"DNN_tsf_{tag}_fold_{fold}"
    with open(f"{output_dir}/models/{tsf_name}.pkl", "wb") as f:
        pickle.dump(tsf, f, protocol=-1)
    joblib.dump(tsf, f"{output_dir}/models/{tsf_name}.joblib")

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
        "model_variant": MODEL_VARIANT,
        "task_test_auc": float(final_test_auc),
        "domain_test_auc": float(domain_test_auc),
        "best_val_auc": float(best_auc),
        "best_selection_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "model_path": best_model_path,
        "tsf_path": f"{output_dir}/models/{tsf_name}.pkl",
        "feature_importance_path": feature_importance_path,
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

if args.fold is not None:
    if args.fold < 1 or args.fold > n_folds:
        raise RuntimeError(f"--fold must be between 1 and {n_folds}, got {args.fold}")
    folds_to_run = [args.fold]
else:
    folds_to_run = list(range(1, n_folds + 1))

for fold in folds_to_run:
    fold_summaries_this = []
    print("\n" + "=" * 90, flush=True)
    print(f"Training fold {fold}", flush=True)
    print("=" * 90, flush=True)

    task_train, task_val, task_test = split_by_fold(task_df, fold, "task_label")
    domain_train, domain_val, domain_test = split_by_fold(domain_df, fold, "domain_label")

    print(f"Fold {fold}: task train/val/test = {len(task_train)}, {len(task_val)}, {len(task_test)}", flush=True)
    print(f"Fold {fold}: domain train/val/test = {len(domain_train)}, {len(domain_val)}, {len(domain_test)}", flush=True)

    scaler = fit_scaler_from_task_train(task_train)
    joblib.dump(scaler, f"{output_dir}/models/scaler_fold_{fold}.pkl")

    task_train_x = transform_df(task_train, scaler, fix_domain_mass=False)
    task_val_x = transform_df(task_val, scaler, fix_domain_mass=False)
    task_test_x = transform_df(task_test, scaler, fix_domain_mass=False)

    domain_train_x = transform_df(domain_train, scaler, fix_domain_mass=True)
    domain_val_x = transform_df(domain_val, scaler, fix_domain_mass=True)
    domain_test_x = transform_df(domain_test, scaler, fix_domain_mass=True)

    arrays_train_common = {
        "task": (
            task_train_x,
            task_train["task_label"].values,
            task_train["train_weight"].values,
        ),
        "domain": (
            domain_train_x,
            domain_train["domain_label"].values,
            domain_train["train_weight"].values,
        ),
    }

    arrays_val_common = {
        "task_X": task_val_x,
        "domain_X": domain_val_x,
    }

    arrays_test_common = {
        "task_X": task_test_x,
        "domain_X": domain_test_x,
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
            arrays_train=arrays_train_common,
            arrays_val=arrays_val_common,
            arrays_test=arrays_test_common,
            dfs_val_test=dfs_val_test_common,
        )
        fold_summaries.append(summary_no)
        fold_summaries_this.append(summary_no)

    if TRAIN_DOMAIN:
        summary_da = train_variant_for_fold(
            fold=fold,
            tag="DA",
            use_domain=True,
            arrays_train=arrays_train_common,
            arrays_val=arrays_val_common,
            arrays_test=arrays_test_common,
            dfs_val_test=dfs_val_test_common,
        )
        fold_summaries.append(summary_da)
        fold_summaries_this.append(summary_da)

    with open(f"{output_dir}/fold_summary_fold_{fold}.json", "w") as f:
        json.dump(to_jsonable(fold_summaries_this), f, indent=2)
    if args.fold is None:
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
    "MODEL_VARIANT": MODEL_VARIANT,
    "ONLY_YEARS": ONLY_YEARS,
    "folds_to_run": folds_to_run,
    "use_source_year": args.use_source_year,
    "no_era_features": args.no_era_features,
    "n_folds": n_folds,
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "dropout": dropout,
    "l2_reg": l2_reg,
    "weight_decay": weight_decay,
    "clipnorm": clipnorm,
    "early_stopping_patience": args.early_stopping_patience,
    "min_delta": args.min_delta,
    "alpha": alpha,
    "max_lambda": max_lambda,
    "focus_weight_max": focus_weight_max,
    "warmup_epochs": warmup_epochs,
    "domain_auc_penalty": domain_auc_penalty,
    "feature_importance_repeats": args.feature_importance_repeats,
    "feature_importance_max_events": args.feature_importance_max_events,
    "FIX_DOMAIN_MASS_TO_125": FIX_DOMAIN_MASS_TO_125,
    "DOMAIN_FIXED_MASS_VALUE": DOMAIN_FIXED_MASS_VALUE,
    "weight_treatment": {
        "task": "signed physical weights; BCE normalized by sum(abs(weight))",
        "domain": "positive abs physical weights; per-class normalized",
    },
    "model": {
        "description": "single shared DNN task classifier; optional GRL domain head",
        "task_input": "all_feature_branches",
        "domain_input": "all_feature_branches, with mass fixed to DOMAIN_FIXED_MASS_VALUE when configured",
        "removed": "all branch1/branch2/branch3/branch4 auxiliary towers",
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
