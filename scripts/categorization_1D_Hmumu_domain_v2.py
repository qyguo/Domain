#!/usr/bin/env python3

import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import uproot
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# Example:
# python3 scripts/categorization_1D_Hmumu_domain_v2.py --score dnn_t \
#   --nscan 100 --nbin 5 --minN 10 --estimate fullSim \
#   --output categorization_all_years

def convert_np(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ============================================================
# User configuration
# ============================================================

CURRENT_DATE = datetime.now().strftime("%m%d")

ALL_ERAS = ["2022", "2022EE", "2023", "2023BPix", "2024", "2025"]

# BASE_PATHS below already point to the DNN application directories. Set this
# to a subdirectory template only when the ROOT files live one level deeper.
INPUT_SUBDIR_TEMPLATE = ""

MASS_BRANCH_BY_ERA = {
    "2022": "diMufsr_rc_BSC_mass",
    "2022EE": "diMufsr_rc_BSC_mass",
    "2023": "diMufsr_rc_BSC_mass",
    "2023BPix": "diMufsr_rc_BSC_mass",
    "2024": "diMufsr_kit_BSC_mass",
    "2025": "diMufsr_kit_BSC_mass",
}

BASE_PATHS = {
    "2022": "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2022_ggHVBF/simpleDNN_DA_0721_2022/",
    "2022EE": "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2022EE_ggHVBF/simpleDNN_DA_0721_2022EE/",
    "2023": "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2023_ggHVBF/simpleDNN_DA_0721_2023/",
    "2023BPix": "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2023BPix_ggHVBF/simpleDNN_DA_0721_2023BPix",
    #"2024": "/eos/user/z/zhangxu/sharing/hmm/2024_v4/skimmed_ntuples/SRSB_v2/",
    #"2025": "/eos/user/h/hakou/Hmumu_Share/qguo/2025/skimmed_ntuples/SRSB_noJetHornVeto/",
    "2024": "/eos/user/h/hakou/Hmumu_Share/qguo/2024_v5_SRSB/skimmed_ntuples/SRSB/simpleDNN_DA_0721_2024/",
    "2025": "/eos/user/h/hakou/Hmumu_Share/qguo/2025_v5_SRSB_v2/skimmed_ntuples/SRSB/simpleDNN_DA_0721_2025/",
}

TREE_NAME_BY_ERA = {
    "2022": "data_two_jet_m110To150_VBF",
    "2022EE": "data_two_jet_m110To150_VBF",
    "2023": "data_two_jet_m110To150_VBF",
    "2023BPix": "data_two_jet_m110To150_VBF",
    "2024": "data_two_jet_m110To150_VBF",
    "2025": "data_two_jet_m110To150_VBF",
}

SAMPLES_BY_ERA = {
    "2022": {
        "data": "data.root",
        "dy_inc": "DY_105To160.root",
        "dy_inc_fail": "DY_105To160_Inc_failvbffilter.root",
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
        "ST_tW_antitop": "ST_tW_antitop.root",
        "ST_tW_top": "ST_tW_top.root",
    },
    "2022EE": {
        "data": "data.root",
        "dy_inc": "DY_105To160.root",
        "dy_inc_fail": "DY_105To160_Inc_failvbffilter.root",
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
        "ST_tW_antitop": "ST_tW_antitop.root",
        "ST_tW_top": "ST_tW_top.root",
    },
    "2023": {
        "data": "data.root",
        "dy_inc": "DY_105To160.root",
        "dy_inc_fail": "DY_105To160_Inc_failvbffilter.root",
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
        "ST_tW_antitop": "ST_tW_antitop.root",
        "ST_tW_top": "ST_tW_top.root",
    },
    "2023BPix": {
        "data": "data.root",
        "dy_inc": "DY_105To160.root",
        "dy_inc_fail": "DY_105To160_Inc_failvbffilter.root",
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
        "ST_tW_antitop": "ST_tW_antitop.root",
        "ST_tW_top": "ST_tW_top.root",
    },
    "2024": {
        "data": "data.root",
        "dy_inc": "DY_105To160.root",
        "dy_inc_fail": "DY_105To160_Inc_failvbffilter.root",
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
        "ST_tW_antitop": "ST_tW_antitop.root",
        "ST_tW_top": "ST_tW_top.root",
    },
    "2025": {
        "data": "data_25_all.root",
        "dy_inc": "DY_105To160.root",
        "dy_inc_fail": "DY_105To160_Inc_failvbffilter.root",
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
        "ST_tW_antitop": "ST_tW_antitop.root",
        "ST_tW_top": "ST_tW_top.root",
    },
}

# Main signal used to optimize VBF category.
SIGNAL_SAMPLES = [
    "vbf_hmm",
]

# Total signal for purity calculation.
SIGNAL_TOTAL_SAMPLES = [
    "vbf_hmm",
    "ggh_hmm",
]

BACKGROUND_SAMPLES = [
    #"dy_inc",
    "dy_inc_fail",
    "dy_vbffilter",
    "ewk_zjj",
    "TTTo2L2Nu",
    "WWTo2L2Nu",
    #"WWW",
    #"WWZ",
    "WZTo2L2Q",
    "WZTo3LNu",
    #"WZZ",
    "ZZTo2L2Nu",
    "ZZTo2L2Q",
    "ZZTo4L",
    #"ZZZ",
    "ST_tW_antitop",
    "ST_tW_top",
]

DATA_SAMPLE = "data"


# ============================================================
# Arguments
# ============================================================

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--years",
        nargs="+",
        default=ALL_ERAS,
        choices=ALL_ERAS,
        help="Eras optimized together. The default uses all six eras.",
    )

    parser.add_argument(
        "--input-subdir-template",
        default=INPUT_SUBDIR_TEMPLATE,
        help="DNN output subdirectory template, e.g. dnn_noDomain_4branch_0611_{era}",
    )

    parser.add_argument(
        "--score",
        default="dnn_t",
        help="Score branch, e.g. dnn_t or dnn_score",
    )

    parser.add_argument(
        "--mass",
        default="diMufsr_kit_BSC_mass",
        help="Mass branch",
    )

    parser.add_argument(
        "--weight",
        default="eventWeight",
        help="Weight branch",
    )

    parser.add_argument("--mass-low", type=float, default=110.0)
    parser.add_argument("--mass-high", type=float, default=150.0)
    parser.add_argument("--sr-low", type=float, default=115.0)
    parser.add_argument("--sr-high", type=float, default=135.0)

    parser.add_argument(
        "--tree",
        default=None,
        help="Override tree name. If None, use TREE_NAME_BY_ERA.",
    )

    parser.add_argument(
        "--nscan",
        type=int,
        default=100,
        help="Number of score scan bins",
    )

    parser.add_argument(
        "--nbin",
        type=int,
        default=5,
        help="Number of final categories",
    )

    parser.add_argument(
        "--minN",
        type=float,
        default=10.0,
        help="Minimum expected background events per category",
    )

    parser.add_argument(
        "--estimate",
        choices=["fullSim", "fullSimrw", "data_sid"],
        default="fullSim",
        help="Background estimate method",
    )

    parser.add_argument(
        "--data-sideband-scale",
        type=float,
        default=None,
        help="Override SR/sideband width scaling in data_sid mode.",
    )

    parser.add_argument(
        "--background-shape",
        choices=["raw", "gaussian", "monotonic"],
        default="monotonic",
        help=(
            "Background score shape used only for boundary optimization. "
            "monotonic applies Gaussian smoothing followed by a non-increasing fit."
        ),
    )

    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.5,
        help="Gaussian smoothing width in score-scan bins.",
    )

    parser.add_argument(
        "--max-background-ratio",
        type=float,
        default=5.0,
        help="Largest allowed B(low-score)/B(next category) for adjacent categories.",
    )

    parser.add_argument(
        "--allow-nonfalling-background",
        action="store_true",
        help="Disable the smooth-falling category-yield constraint.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output directory",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information",
    )

    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================

def asimov_z(s, b):
    s = float(max(s, 0.0))
    b = float(max(b, 0.0))

    if s <= 0.0:
        return 0.0

    if b <= 0.0:
        return np.sqrt(2.0 * s)

    return np.sqrt(max(0.0, 2.0 * ((s + b) * np.log(1.0 + s / b) - s)))


def calc_category_z(s, b):
    return asimov_z(s, b)


def get_input_file(era, sample_key, args):
    subdir = args.input_subdir_template.format(era=era).strip()
    folder = os.path.join(BASE_PATHS[era], subdir) if subdir else BASE_PATHS[era]
    filename = SAMPLES_BY_ERA[era][sample_key]
    return os.path.join(folder, filename)


def get_tree_name(era, override=None):
    if override is not None:
        return override
    return TREE_NAME_BY_ERA[era]


def available_branches(root_file, tree_name):
    with uproot.open(root_file) as f:
        if tree_name not in f:
            raise RuntimeError(
                f"Tree {tree_name} not found in {root_file}. Available keys: {list(f.keys())}"
            )
        return set(f[tree_name].keys())


def resolve_mass_branch(era, requested, available):
    preferred = MASS_BRANCH_BY_ERA[era]
    candidates = [preferred, requested]

    if "_kit_BSC_" in requested:
        candidates.append(requested.replace("_kit_BSC_", "_rc_BSC_"))
    elif "_rc_BSC_" in requested:
        candidates.append(requested.replace("_rc_BSC_", "_kit_BSC_"))

    for candidate in dict.fromkeys(candidates):
        if candidate in available:
            return candidate

    raise RuntimeError(
        f"No mass branch found for era={era}. Tried {list(dict.fromkeys(candidates))}"
    )


def read_sample(era, sample_key, args):
    path = get_input_file(era, sample_key, args)
    tree_name = get_tree_name(era, args.tree)

    if not os.path.exists(path):
        print(f"[WARNING] Missing file, skip: {path}", flush=True)
        return pd.DataFrame()

    branches = available_branches(path, tree_name)

    mass_input = resolve_mass_branch(era, args.mass, branches)
    required = [args.score, mass_input, args.weight]

    read_cols = []
    for c in required:
        if c in branches:
            read_cols.append(c)

    if args.score not in read_cols:
        raise RuntimeError(f"Score branch {args.score} not found in {path}")

    if args.weight not in read_cols:
        print(f"[WARNING] Weight branch {args.weight} not found in {path}. Use weight=1.", flush=True)

    print(f"[INFO] Read {era} {sample_key}: {path}", flush=True)

    with uproot.open(path) as f:
        df = f[tree_name].arrays(read_cols, library="pd")

    if mass_input != args.mass:
        df = df.rename(columns={mass_input: args.mass})
        print(f"[INFO] Branch alias for {era}: {args.mass} <- {mass_input}", flush=True)

    df["era"] = era
    df["sample"] = sample_key

    if args.weight not in df.columns:
        df[args.weight] = 1.0

    if sample_key == DATA_SAMPLE:
        df[args.weight] = 1.0

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[args.score, args.mass, args.weight])

    df[args.score] = df[args.score].astype(float)
    df[args.mass] = df[args.mass].astype(float)
    df[args.weight] = df[args.weight].astype(float)

    df = df[(df[args.score] >= 0.0) & (df[args.score] <= 1.0)].copy()

    print(f"[INFO] Loaded {len(df)} events after cleanup.", flush=True)

    return df


def load_samples(years, sample_keys, args):
    dfs = []

    for era in years:
        if era not in BASE_PATHS:
            print(f"[WARNING] Unknown era {era}, skip.", flush=True)
            continue

        for sample_key in sample_keys:
            if sample_key not in SAMPLES_BY_ERA[era]:
                print(f"[WARNING] {sample_key} not configured for {era}, skip.", flush=True)
                continue

            df = read_sample(era, sample_key, args)
            if len(df) > 0:
                dfs.append(df)

    if len(dfs) == 0:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def validate_loaded_eras(dfs, args):
    required_groups = ["sig", "bkg"]
    if args.estimate in ["data_sid", "fullSimrw"]:
        required_groups.append("data")

    for group in required_groups:
        frame = dfs[group]
        loaded = set(frame["era"].astype(str).unique()) if len(frame) > 0 else set()
        missing = [era for era in args.years if era not in loaded]
        if missing:
            raise RuntimeError(
                f"Cannot perform the combined optimization: {group} has no loaded events "
                f"for eras {missing}. Check BASE_PATHS, filenames, tree names, and DNN outputs."
            )


def apply_mass_region(df, args, region):
    if len(df) == 0:
        return df.copy()

    m = df[args.mass].values

    if region == "sr":
        return df[(m >= args.sr_low) & (m <= args.sr_high)].copy()

    if region == "sid":
        return df[
            (m >= args.mass_low)
            & (m <= args.mass_high)
            & ((m < args.sr_low) | (m > args.sr_high))
        ].copy()

    if region == "tot":
        return df[(m >= args.mass_low) & (m <= args.mass_high)].copy()

    raise ValueError(f"Unknown region: {region}")


def hist_score(df, args, nscan):
    if len(df) == 0:
        return np.zeros(nscan), np.zeros(nscan)

    score = df[args.score].values
    weight = df[args.weight].values

    hist, edges = np.histogram(
        score,
        bins=nscan,
        range=(0.0, 1.0),
        weights=weight,
    )

    hist_w2, _ = np.histogram(
        score,
        bins=nscan,
        range=(0.0, 1.0),
        weights=weight * weight,
    )

    return hist.astype(float), hist_w2.astype(float)


def gaussian_smooth(hist, sigma):
    values = np.clip(np.asarray(hist, dtype=float), 0.0, None)
    total = values.sum()
    if sigma <= 0.0 or len(values) < 2 or total <= 0.0:
        return values

    radius = max(1, int(np.ceil(4.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    padded = np.pad(values, radius, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    smoothed = np.clip(smoothed, 0.0, None)
    if smoothed.sum() > 0.0:
        smoothed *= total / smoothed.sum()
    return smoothed


def isotonic_decreasing(values):
    """Unweighted pool-adjacent-violators fit constrained to be decreasing."""
    values = np.asarray(values, dtype=float)
    blocks = []

    for index, value in enumerate(values):
        blocks.append([index, index + 1, 1.0, float(value)])
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = left[3] / left[2]
            right_mean = right[3] / right[2]
            if left_mean >= right_mean:
                break
            blocks[-2:] = [[left[0], right[1], left[2] + right[2], left[3] + right[3]]]

    fitted = np.empty_like(values)
    for start, end, weight, total in blocks:
        fitted[start:end] = total / weight
    return np.clip(fitted, 0.0, None)


def regularize_background(hist, mode, sigma):
    input_values = np.asarray(hist, dtype=float)
    signed_total = float(input_values.sum())
    n_negative = int(np.sum(input_values < 0.0))
    raw = np.clip(input_values, 0.0, None)
    if signed_total > 0.0 and raw.sum() > 0.0:
        raw *= signed_total / raw.sum()
    if mode == "raw":
        result = raw
    else:
        result = gaussian_smooth(raw, sigma)
        if mode == "monotonic":
            result = isotonic_decreasing(result)

    if raw.sum() > 0.0 and result.sum() > 0.0:
        result *= raw.sum() / result.sum()

    rising_steps = int(np.sum(np.diff(result) > 1e-12))
    print(
        f"[INFO] Background shape={mode}: raw integral={raw.sum():.6g}, "
        f"optimization integral={result.sum():.6g}, negative raw bins={n_negative}, "
        f"rising steps={rising_steps}",
        flush=True,
    )
    return result


def build_histograms(dfs, args, nscan):
    sig_sr = apply_mass_region(dfs["sig"], args, "sr")
    sig_tot_sr = apply_mass_region(dfs["sig_tot"], args, "sr")
    bkg_sr = apply_mass_region(dfs["bkg"], args, "sr")
    bkg_sid = apply_mass_region(dfs["bkg"], args, "sid")

    h_sig, h_sig_w2 = hist_score(sig_sr, args, nscan)
    h_sig_tot, h_sig_tot_w2 = hist_score(sig_tot_sr, args, nscan)
    h_bkg_sr, h_bkg_sr_w2 = hist_score(bkg_sr, args, nscan)
    h_bkg_sid, h_bkg_sid_w2 = hist_score(bkg_sid, args, nscan)

    if "data" in dfs and len(dfs["data"]) > 0:
        data_sid = apply_mass_region(dfs["data"], args, "sid")
        h_data_sid, h_data_sid_w2 = hist_score(data_sid, args, nscan)
    else:
        h_data_sid = np.zeros(nscan)
        h_data_sid_w2 = np.zeros(nscan)

    if args.estimate == "fullSim":
        h_bkg_est = h_bkg_sr.copy()
        h_bkg_est_w2 = h_bkg_sr_w2.copy()

    elif args.estimate == "data_sid":
        sideband_width = (
            (args.sr_low - args.mass_low)
            + (args.mass_high - args.sr_high)
        )
        if sideband_width <= 0.0:
            raise RuntimeError("The configured sideband has non-positive width.")
        scale = args.data_sideband_scale
        if scale is None:
            scale = (args.sr_high - args.sr_low) / sideband_width
        h_bkg_est = scale * h_data_sid
        h_bkg_est_w2 = (scale ** 2) * h_data_sid_w2

    elif args.estimate == "fullSimrw":
        ratio = np.divide(
            h_data_sid,
            h_bkg_sid,
            out=np.ones_like(h_data_sid),
            where=(h_bkg_sid > 0),
        )

        h_bkg_est = h_bkg_sr * ratio

        rel2_sr = np.divide(
            h_bkg_sr_w2,
            h_bkg_sr ** 2,
            out=np.zeros_like(h_bkg_sr),
            where=(h_bkg_sr > 0),
        )
        rel2_data = np.divide(
            h_data_sid_w2,
            h_data_sid ** 2,
            out=np.zeros_like(h_data_sid),
            where=(h_data_sid > 0),
        )
        rel2_sid = np.divide(
            h_bkg_sid_w2,
            h_bkg_sid ** 2,
            out=np.zeros_like(h_bkg_sid),
            where=(h_bkg_sid > 0),
        )

        h_bkg_est_w2 = (h_bkg_est ** 2) * (rel2_sr + rel2_data + rel2_sid)

    else:
        raise RuntimeError(f"Unknown estimate {args.estimate}")

    h_bkg_for_optimization = regularize_background(
        h_bkg_est,
        mode=args.background_shape,
        sigma=args.smooth_sigma,
    )

    return {
        "sig": h_sig,
        "sig_w2": h_sig_w2,
        "sig_tot": h_sig_tot,
        "sig_tot_w2": h_sig_tot_w2,
        "bkg": h_bkg_for_optimization,
        "bkg_w2": h_bkg_est_w2,
        "bkg_raw": h_bkg_est,
        "bkg_raw_w2": h_bkg_est_w2,
        "bkgmc_sr": h_bkg_sr,
        "bkgmc_sr_w2": h_bkg_sr_w2,
        "bkgmc_sid": h_bkg_sid,
        "bkgmc_sid_w2": h_bkg_sid_w2,
        "data_sid": h_data_sid,
        "data_sid_w2": h_data_sid_w2,
    }


# ============================================================
# Boundary optimization
# ============================================================

def interval_sum(arr_cum, start, end):
    return arr_cum[end] - arr_cum[start]


def optimize_boundaries_from_hist(
    h_sig,
    h_bkg,
    nbin,
    minN,
    require_falling=True,
    max_background_ratio=5.0,
):
    nscan = len(h_sig)

    sig_cum = np.concatenate([[0.0], np.cumsum(h_sig)])
    bkg_cum = np.concatenate([[0.0], np.cumsum(h_bkg)])

    if not require_falling:
        dp = np.full((nbin + 1, nscan + 1), -np.inf)
        prev = np.full((nbin + 1, nscan + 1), -1, dtype=int)
        dp[0, 0] = 0.0

        for k in range(1, nbin + 1):
            for end in range(k, nscan + 1):
                for start in range(k - 1, end):
                    if not np.isfinite(dp[k - 1, start]):
                        continue
                    s = interval_sum(sig_cum, start, end)
                    b = interval_sum(bkg_cum, start, end)
                    if b < minN:
                        continue
                    value = dp[k - 1, start] + calc_category_z(s, b) ** 2
                    if value > dp[k, end]:
                        dp[k, end] = value
                        prev[k, end] = start

        best_value = dp[nbin, nscan]
        boundaries_bins = [nscan]
        k = nbin
        end = nscan
        while k > 0:
            start = prev[k, end]
            if start < 0:
                break
            boundaries_bins.append(start)
            end = start
            k -= 1
    else:
        # State (k, start, end) stores the best k-category solution whose last
        # category is [start, end). Retaining the previous interval allows an
        # explicit constraint on adjacent category background yields.
        dp = np.full((nbin + 1, nscan + 1, nscan + 1), -np.inf)
        prev_start = np.full((nbin + 1, nscan + 1, nscan + 1), -1, dtype=np.int32)

        for end in range(1, nscan + 1):
            s = interval_sum(sig_cum, 0, end)
            b = interval_sum(bkg_cum, 0, end)
            if b >= minN:
                dp[1, 0, end] = calc_category_z(s, b) ** 2

        for k in range(2, nbin + 1):
            for end in range(k, nscan + 1):
                for start in range(k - 1, end):
                    s_now = interval_sum(sig_cum, start, end)
                    b_now = interval_sum(bkg_cum, start, end)
                    if b_now < minN:
                        continue

                    for previous_start in range(k - 2, start):
                        previous_value = dp[k - 1, previous_start, start]
                        if not np.isfinite(previous_value):
                            continue

                        b_previous = interval_sum(bkg_cum, previous_start, start)
                        if b_previous + 1e-12 < b_now:
                            continue
                        if b_previous > max_background_ratio * b_now + 1e-12:
                            continue

                        value = previous_value + calc_category_z(s_now, b_now) ** 2
                        if value > dp[k, start, end]:
                            dp[k, start, end] = value
                            prev_start[k, start, end] = previous_start

        last_start = int(np.argmax(dp[nbin, :, nscan]))
        best_value = dp[nbin, last_start, nscan]
        boundaries_bins = [nscan, last_start]
        k = nbin
        end = nscan
        start = last_start

        while k > 1 and np.isfinite(best_value):
            previous = int(prev_start[k, start, end])
            if previous < 0:
                break
            boundaries_bins.append(previous)
            end = start
            start = previous
            k -= 1

    if not np.isfinite(best_value) or len(boundaries_bins) != nbin + 1:
        constraint = (
            f", falling background and max adjacent ratio={max_background_ratio}"
            if require_falling
            else ""
        )
        raise RuntimeError(
            f"Could not find {nbin} categories with minN={minN}{constraint}. "
            "Try fewer categories, smaller minN, a larger --max-background-ratio, "
            "or --allow-nonfalling-background."
        )

    boundaries_bins = sorted(boundaries_bins)
    boundaries_values = [b / float(nscan) for b in boundaries_bins[:-1]]
    total_z = np.sqrt(best_value)

    return boundaries_bins, boundaries_values, total_z


def evaluate_boundaries(hists, boundaries_bins):
    h_sig = hists["sig"]
    h_sig_w2 = hists["sig_w2"]
    h_sig_tot = hists["sig_tot"]
    h_sig_tot_w2 = hists["sig_tot_w2"]
    h_bkg = hists["bkg"]
    h_bkg_w2 = hists["bkg_w2"]
    h_bkg_raw = hists["bkg_raw"]
    h_bkg_raw_w2 = hists["bkg_raw_w2"]
    h_bkgmc_sr = hists["bkgmc_sr"]
    h_bkgmc_sr_w2 = hists["bkgmc_sr_w2"]
    h_bkgmc_sid = hists["bkgmc_sid"]
    h_bkgmc_sid_w2 = hists["bkgmc_sid_w2"]
    h_data_sid = hists["data_sid"]
    h_data_sid_w2 = hists["data_sid_w2"]

    rows = []

    for i in range(len(boundaries_bins) - 1):
        lo = boundaries_bins[i]
        hi = boundaries_bins[i + 1]

        sig = h_sig[lo:hi].sum()
        sig_err = np.sqrt(h_sig_w2[lo:hi].sum())

        sig_tot = h_sig_tot[lo:hi].sum()
        sig_tot_err = np.sqrt(h_sig_tot_w2[lo:hi].sum())

        bkg = h_bkg[lo:hi].sum()
        bkg_err = np.sqrt(h_bkg_w2[lo:hi].sum())

        bkg_raw = h_bkg_raw[lo:hi].sum()
        bkg_raw_err = np.sqrt(h_bkg_raw_w2[lo:hi].sum())

        bkgmc_sr = h_bkgmc_sr[lo:hi].sum()
        bkgmc_sr_err = np.sqrt(h_bkgmc_sr_w2[lo:hi].sum())

        bkgmc_sid = h_bkgmc_sid[lo:hi].sum()
        bkgmc_sid_err = np.sqrt(h_bkgmc_sid_w2[lo:hi].sum())

        data_sid = h_data_sid[lo:hi].sum()
        data_sid_err = np.sqrt(h_data_sid_w2[lo:hi].sum())

        z = calc_category_z(sig, bkg)

        purity = 100.0 * sig / sig_tot if sig_tot > 0 else 0.0

        rows.append({
            "cat": i,
            "score_low": lo / float(len(h_sig)),
            "score_high": hi / float(len(h_sig)),
            "sig": sig,
            "sig_err": sig_err,
            "sig_tot": sig_tot,
            "sig_tot_err": sig_tot_err,
            "bkg": bkg,
            "bkg_err": bkg_err,
            "bkg_raw": bkg_raw,
            "bkg_raw_err": bkg_raw_err,
            "bkgmc_sr": bkgmc_sr,
            "bkgmc_sr_err": bkgmc_sr_err,
            "bkgmc_sid": bkgmc_sid,
            "bkgmc_sid_err": bkgmc_sid_err,
            "data_sid": data_sid,
            "data_sid_err": data_sid_err,
            "z": z,
            "VBF_purity_percent": purity,
        })

    out = pd.DataFrame(rows)
    total_z = np.sqrt(np.sum(out["z"].values ** 2))

    return out, total_z


def evaluate_each_era(dfs, args, boundaries_bins):
    tables = []
    for era in args.years:
        era_dfs = {
            key: df[df["era"] == era].copy() if len(df) > 0 else df.copy()
            for key, df in dfs.items()
        }
        if len(era_dfs["sig"]) == 0 and len(era_dfs["bkg"]) == 0:
            continue
        era_hists = build_histograms(era_dfs, args, nscan=args.nscan)
        era_yields, era_z = evaluate_boundaries(era_hists, boundaries_bins)
        era_yields.insert(0, "era", era)
        era_yields["combined_boundary_significance"] = era_z
        tables.append(era_yields)

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def plot_diagnostics(hists, yields, boundaries_bins, args):
    edges = np.linspace(0.0, 1.0, args.nscan + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    ax.step(centers, np.clip(hists["bkg_raw"], 0.0, None), where="mid", color="0.45", label="Raw background")
    ax.plot(centers, hists["bkg"], color="#0072B2", linewidth=2.0, label="Optimization background")
    for boundary in boundaries_bins[1:-1]:
        ax.axvline(boundary / float(args.nscan), color="#D55E00", linewidth=0.9, alpha=0.75)
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=max(1e-3, np.min(hists["bkg"][hists["bkg"] > 0]) * 0.5))
    ax.set_xlabel(args.score)
    ax.set_ylabel("Expected background / scan bin")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(os.path.join(args.output, f"background_shape_and_boundaries.{extension}"), dpi=160)
    plt.close(fig)

    cats = np.arange(len(yields))
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    raw_for_plot = np.clip(yields["bkg_raw"].to_numpy(dtype=float), 1e-6, None)
    ax.errorbar(
        cats,
        raw_for_plot,
        yerr=yields["bkg_raw_err"].to_numpy(dtype=float),
        fmt="o",
        color="0.25",
        capsize=2,
        label="Raw background",
    )
    ax.plot(cats, yields["bkg"], "s-", color="#0072B2", label="Optimization background")
    ax.set_yscale("log")
    ax.set_xticks(cats)
    ax.set_xlabel("Category (low to high DNN score)")
    ax.set_ylabel("Expected background in SR")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(os.path.join(args.output, f"category_background_yields.{extension}"), dpi=160)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    args = get_args()
    args.years = list(dict.fromkeys(args.years))

    if not (args.mass_low < args.sr_low < args.sr_high < args.mass_high):
        raise RuntimeError(
            "Require mass_low < sr_low < sr_high < mass_high; got "
            f"{args.mass_low}, {args.sr_low}, {args.sr_high}, {args.mass_high}."
        )
    if args.max_background_ratio < 1.0:
        raise RuntimeError("--max-background-ratio must be at least 1.")
    if args.nbin < 1 or args.nbin > args.nscan:
        raise RuntimeError("Require 1 <= nbin <= nscan.")

    if args.output is None:
        args.output = f"categorization_outputs_{CURRENT_DATE}"

    os.makedirs(args.output, exist_ok=True)

    print("[INFO] Inclusive 1D categorization")
    print("[INFO] No fold split is used")
    print("[INFO] Years:", args.years)
    print("[INFO] Input subdir template:", args.input_subdir_template)
    print("[INFO] Score:", args.score)
    print("[INFO] Estimate:", args.estimate)
    print("[INFO] nscan:", args.nscan)
    print("[INFO] nbin:", args.nbin)
    print("[INFO] minN:", args.minN)
    print("[INFO] SR:", (args.sr_low, args.sr_high))
    print("[INFO] Background shape:", args.background_shape)
    print("[INFO] Smooth sigma:", args.smooth_sigma)
    print("[INFO] Require falling category background:", not args.allow_nonfalling_background)
    print("[INFO] Max adjacent background ratio:", args.max_background_ratio)
    print("[INFO] Output:", args.output)

    print("\n[INFO] Loading signal samples")
    df_sig = load_samples(args.years, SIGNAL_SAMPLES, args)

    print("\n[INFO] Loading total signal samples")
    df_sig_tot = load_samples(args.years, SIGNAL_TOTAL_SAMPLES, args)

    print("\n[INFO] Loading background samples")
    df_bkg = load_samples(args.years, BACKGROUND_SAMPLES, args)

    print("\n[INFO] Loading data sample")
    df_data = load_samples(args.years, [DATA_SAMPLE], args)

    dfs = {
        "sig": df_sig,
        "sig_tot": df_sig_tot,
        "bkg": df_bkg,
        "data": df_data,
    }
    validate_loaded_eras(dfs, args)

    print("\n[INFO] Sample sizes:")
    for key, df in dfs.items():
        print(f"  {key:8s}: {len(df)}")

    print("\n" + "=" * 100)
    print("[INFO] Optimize inclusive boundaries")
    print("=" * 100)

    hists = build_histograms(
        dfs,
        args,
        nscan=args.nscan,
    )

    boundaries_bins, boundaries_values, smax = optimize_boundaries_from_hist(
        hists["sig"],
        hists["bkg"],
        nbin=args.nbin,
        minN=args.minN,
        require_falling=not args.allow_nonfalling_background,
        max_background_ratio=args.max_background_ratio,
    )

    yields, total_z = evaluate_boundaries(
        hists,
        boundaries_bins,
    )

    print("\n[INFO] Final boundaries bins:")
    print(boundaries_bins)

    print("\n[INFO] Final boundaries values:")
    print(boundaries_values)

    print("\n[INFO] Yields:")
    print(yields)

    if len(yields) > 1:
        bkg_values = yields["bkg"].to_numpy(dtype=float)
        ratios = np.divide(
            bkg_values[:-1],
            bkg_values[1:],
            out=np.full(len(bkg_values) - 1, np.inf),
            where=bkg_values[1:] > 0.0,
        )
        print("\n[INFO] Adjacent optimization-background ratios B[i]/B[i+1]:")
        print(ratios)

    print(f"\n[INFO] Optimization significance: {smax:.4f}")
    print(f"[INFO] Final evaluated significance: {total_z:.4f}")

    outs = {
        "boundaries_bins": boundaries_bins,
        "boundaries_values": boundaries_values,
        "smax": smax,
        "significance": total_z,
        "nscan": args.nscan,
        "nbin": args.nbin,
        "minN": args.minN,
        "score": args.score,
        "mass": args.mass,
        "weight": args.weight,
        "estimate": args.estimate,
        "data_sideband_scale": (
            args.data_sideband_scale
            if args.data_sideband_scale is not None
            else (args.sr_high - args.sr_low)
            / ((args.sr_low - args.mass_low) + (args.mass_high - args.sr_high))
        ),
        "background_shape": args.background_shape,
        "smooth_sigma": args.smooth_sigma,
        "require_falling_background": not args.allow_nonfalling_background,
        "max_background_ratio": args.max_background_ratio,
        "mass_window": [args.mass_low, args.mass_high],
        "sr_window": [args.sr_low, args.sr_high],
        "years": args.years,
        "input_subdir_template": args.input_subdir_template,
        "inclusive": True,
    }

    json_path = os.path.join(
        args.output,
        f"categorization_inclusive_{args.score}_{args.nbin}cat_{args.minN}minN.json",
    )
    yields_path = os.path.join(
        args.output,
        f"categorization_inclusive_yields_{args.score}_{args.nbin}cat_{args.minN}minN.csv",
    )
    era_yields_path = os.path.join(
        args.output,
        f"categorization_by_era_yields_{args.score}_{args.nbin}cat_{args.minN}minN.csv",
    )
    txt_path = os.path.join(
        args.output,
        f"bin_boundaries_1D_inclusive_{args.score}_{args.nbin}cat_{args.minN}minN.txt",
    )

    with open(json_path, "w") as f:
        json.dump(outs, f, indent=2, default=convert_np)
        
    yields.to_csv(yields_path, index=False)

    era_yields = evaluate_each_era(dfs, args, boundaries_bins)
    era_yields.to_csv(era_yields_path, index=False)

    plot_diagnostics(hists, yields, boundaries_bins, args)

    with open(txt_path, "w") as f:
        f.write("1\n")
        f.write("1\n")
        f.write(f"{len(boundaries_values)} ")
        for v in boundaries_values:
            f.write(f"{v:.4f} ")
        f.write("1.0000\n")
        for z in yields["z"].values:
            f.write(f"{z:.4f} ")
        f.write(f"{total_z:.4f}\n")

    print("\n[INFO] Wrote:")
    print(" ", json_path)
    print(" ", yields_path)
    print(" ", era_yields_path)
    print(" ", txt_path)
    print(" ", os.path.join(args.output, "background_shape_and_boundaries.png"))
    print(" ", os.path.join(args.output, "category_background_yields.png"))


if __name__ == "__main__":
    main()
