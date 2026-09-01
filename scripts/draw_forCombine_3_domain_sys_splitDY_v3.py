import os
import argparse
import re
import ROOT
import array
import uuid
from tqdm import tqdm


ROOT.gROOT.SetBatch(True)


ALL_ERAS = ["2022", "2022EE", "2023", "2023BPix", "2024", "2025"]

BASE_PATHS = {
    "2022": "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2022_ggHVBF/simpleDNN_DA_0721_2022/",
    "2022EE": "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2022EE_ggHVBF/simpleDNN_DA_0721_2022EE/",
    "2023": "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2023_ggHVBF/simpleDNN_DA_0721_2023/",
    "2023BPix": "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2023BPix_ggHVBF/simpleDNN_DA_0721_2023BPix/",
    ##"2024": "/eos/user/h/hakou/Hmumu_Share/qguo/2024_v5_SRSB/skimmed_ntuples/SRSB/simpleDNN_DA_0721_2024/",
    ##"2025": "/eos/user/h/hakou/Hmumu_Share/qguo/2025_v5_SRSB_v2/skimmed_ntuples/SRSB/simpleDNN_DA_0721_2025/",
    "2024": "/eos/user/h/hakou/Hmumu_Share/qguo/2024_v5_SRSB/skimmed_ntuples/SRSB/simpleDNN_DA_0721_2024/",
    "2025": "/eos/user/h/hakou/Hmumu_Share/qguo/2025_v5_SRSB_v2/skimmed_ntuples/SRSB/simpleDNN_DA_0721_2025/",
    #"2022": "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2022_ggHVBF/simpleDNN_DA_0728_2022/",
    #"2022EE": "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2022EE_ggHVBF/simpleDNN_DA_0728_2022EE/",
    #"2023": "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2023_ggHVBF/simpleDNN_DA_0728_2023/",
    #"2023BPix": "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2023BPix_ggHVBF/simpleDNN_DA_0728_2023BPix/",
    #"2024": "/eos/user/h/hakou/Hmumu_Share/qguo/2024_v5_SRSB/skimmed_ntuples/SRSB/simpleDNN_DA_0728_2024/",
    #"2025": "/eos/user/h/hakou/Hmumu_Share/qguo/2025_v5_SRSB_v2/skimmed_ntuples/SRSB/simpleDNN_DA_0728_2025/",
}

TREE_NAME_BY_ERA = {era: "data_two_jet_m110To150_VBF" for era in ALL_ERAS}

MASS_BRANCH_BY_ERA = {
    "2022": "diMufsr_rc_BSC_mass",
    "2022EE": "diMufsr_rc_BSC_mass",
    "2023": "diMufsr_rc_BSC_mass",
    "2023BPix": "diMufsr_rc_BSC_mass",
    "2024": "diMufsr_kit_BSC_mass",
    "2025": "diMufsr_kit_BSC_mass",
}

# eventWeight is used without an additional luminosity correction by default.
# Put an era-specific correction here only if the ntuple normalization needs it.
MC_SCALE_BY_ERA = {era: 1.0 for era in ALL_ERAS}

BIN_EDGES = [0.0, 0.29, 0.36, 0.42, 0.47, 0.53, 0.59, 0.63, 0.67, 0.71, 0.75, 0.79, 0.83, 0.9, 1.0,]
#BIN_EDGES = [0.0, 0.395, 0.51, 0.595, 0.656, 0.716, 0.767, 0.8, 0.834, 0.867, 0.899, 0.934, 0.968, 0.984,1.0]

DY_SPLIT_SELECTIONS = {
    "DYJ01": "n_jets_matched_genjet <= 1",
    "DYJ2": "n_jets_matched_genjet >= 2",
}


def categories_for_era(era):
    early_era = era in {"2022", "2022EE", "2023", "2023BPix"}
    signal_suffix = "_ggHUnc" if early_era else ""
    dy_inclusive = (
        "DY_105To160.root"
        if early_era
        else "DY_105To160_Inc_failvbffilter.root"
    )
    dy_files = [dy_inclusive, "DY_105To160_Fil-VBF_passvbffilter.root"]
    dy_files=["DY_105To160_Inc_failvbffilter.root", "DY_105To160_Fil-VBF_passvbffilter.root"]

    return {
        "vh_hmm": ["VHToMuMu_M125.root"],
        "tth_hmm": ["TTHto2Mu_M-125.root"],
        "rare_hmm": ["rareHToMuMu_M125.root"],
        "qqH_hmm": [f"VBFHToMuMu_M125{signal_suffix}.root"],
        "ggH_hmm": [f"GluGluHToMuMu_M125{signal_suffix}.root"],
        "DYJ01": dy_files,
        "DYJ2": dy_files,
        "EWKZ": ["EWK_LLJJ_M105To160.root"],
        "Top": ["ST_tW_antitop.root", "ST_tW_top.root", "TTTo2L2Nu.root"],
        "VV": [
            "ZZTo2L2Q.root", "ZZTo2L2Nu.root", "ZZTo4L.root",
            "WZTo3LNu.root", "WZTo2L2Q.root", "WWTo2L2Nu.root",
        ],
        "data_obs": ["data.root"],
    }


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build per-era Combine histograms with separate DYJ01 and DYJ2 "
            "processes from nominal and systematic DNN trees."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--region", choices=["SR", "SB", "sr", "sb"], default="SR")
    parser.add_argument("--year", choices=["all"] + ALL_ERAS, default="all")
    parser.add_argument("--score", default="dnn_t")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument(
        "--tree",
        default=None,
        help="Override the configured nominal tree for every selected era.",
    )
    parser.add_argument(
        "--mc-scale",
        type=float,
        default=None,
        help="Override MC_SCALE_BY_ERA for every selected era.",
    )
    return parser.parse_args()


def build_cut(var, region):
    region = region.upper()
    if region == "SR":
        return f"({var} >= 115 && {var} <= 135)"
    if region == "SB":
        return f"(({var} >= 110 && {var} < 115) || ({var} > 135 && {var} <= 150))"
    raise ValueError("Unknown region: use SR or SB")


def list_matching_trees(tfile, base_tree):
    """Return all TTree names equal to base_tree or starting with base_tree__."""
    out = []
    for key in tfile.GetListOfKeys():
        name = key.GetName()
        obj = tfile.Get(name)
        if not obj:
            continue
        if not obj.InheritsFrom("TTree"):
            continue
        if name == base_tree or name.startswith(base_tree + "__"):
            out.append(name)
    return sorted(out)


def discover_trees_for_files(file_paths, base_tree):
    """Find matching trees that actually exist in these files."""
    found = set()
    for fp in file_paths:
        if not os.path.exists(fp):
            continue
        tf = ROOT.TFile.Open(fp)
        if not tf or tf.IsZombie():
            continue
        for t in list_matching_trees(tf, base_tree):
            found.add(t)
        tf.Close()
    return sorted(found)


def syst_suffix_from_tree(tree_name, base_tree):
    """
    base_tree                  -> ""   (nominal)
    base_tree__jerbarrelUp     -> "jerbarrelUp"
    base_tree__XYZDown         -> "XYZDown"
    """
    if tree_name == base_tree:
        return ""
    prefix = base_tree + "__"
    if tree_name.startswith(prefix):
        return tree_name[len(prefix):]
    return ""


def tree_has_branches(tree, branch_names):
    return all(tree.GetBranch(b) for b in branch_names)


def make_histogram(
    file_paths,
    tree_name,
    var_to_cut,
    var_to_plot,
    hist_name,
    bin_edges,
    weight_expr,
    SR_or_SB,
    scale=1.0,
    fallback_tree_name=None,
    extra_cut=None,
    extra_required_branches=None,
):
    bin_array = array.array("d", bin_edges)
    hist = ROOT.TH1D(hist_name, hist_name, len(bin_edges) - 1, bin_array)
    hist.Sumw2()
    hist.SetDirectory(0)

    required = [var_to_cut, var_to_plot]
    for branch_name in extra_required_branches or []:
        if branch_name not in required:
            required.append(branch_name)
    if weight_expr not in ("1", "1.0"):
        for token in re.findall(r"[A-Za-z_]\w*", weight_expr):
            if token not in required:
                required.append(token)

    for file_path in file_paths:
        print(f"[INFO] Opening file: {file_path}", flush=True)
        tf = ROOT.TFile.Open(file_path)
        if not tf or not tf.IsOpen() or tf.IsZombie():
            print(f"[WARNING] Failed to open file: {file_path}", flush=True)
            continue

        tree = tf.Get(tree_name)
        usable = tree and tree_has_branches(tree, required)
        if not usable and fallback_tree_name and fallback_tree_name != tree_name:
            fallback = tf.Get(fallback_tree_name)
            if fallback and tree_has_branches(fallback, required):
                print(
                    f"[WARNING] Use nominal tree '{fallback_tree_name}' for {file_path} "
                    f"because systematic tree '{tree_name}' is unavailable or incomplete.",
                    flush=True,
                )
                tree = fallback
                usable = True

        if not usable:
            print(
                f"[WARNING] No usable tree for {file_path}: requested '{tree_name}', "
                f"required branches {required}",
                flush=True,
            )
            tf.Close()
            continue

        cut_expr = build_cut(var_to_cut, SR_or_SB)
        if extra_cut:
            cut_expr = f"({cut_expr}) && ({extra_cut})"
        if weight_expr in ("1", "1.0"):
            cut_str = f"({cut_expr})"
        else:
            cut_str = f"({weight_expr})*{scale}*({cut_expr})"

        # Keep temporary histograms outside the input TFile. Let Python own
        # their lifetime; explicit Delete() can double-delete PyROOT objects.
        ROOT.gROOT.cd()
        temp_name = f"tmp_{hist_name}_{uuid.uuid4().hex}"
        temp_hist = ROOT.TH1D(temp_name, "", len(bin_edges) - 1, bin_array)
        temp_hist.Sumw2()

        tree.Draw(f"{var_to_plot} >> {temp_name}", cut_str, "goff")
        hist.Add(temp_hist)
        temp_hist.SetDirectory(0)
        del temp_hist

        tf.Close()

    print(f"[INFO] {hist_name}: yield = {hist.Integral():.6g}", flush=True)
    return hist


def write_hist(fout, hist, name):
    if hist is None:
        return
    hist.SetName(name)
    hist.SetTitle(name)
    fout.cd()
    hist.Write(name, ROOT.TObject.kOverwrite)
    print(f"[INFO] Wrote histogram: {name}", flush=True)


def build_weight_syst_hist(
    file_paths,
    base_tree,
    category,
    syst_name,
    branch,
    direction,
    var_to_cut,
    var_to_plot,
    bin_edges,
    nominal_weight,
    SR_or_SB,
    scale,
    extra_cut=None,
    extra_required_branches=None,
):
    hname = f"{category}_{syst_name}{direction}"
    bin_array = array.array("d", bin_edges)
    total = ROOT.TH1D(hname, hname, len(bin_edges) - 1, bin_array)
    total.Sumw2()
    total.SetDirectory(0)
    n_usable = 0

    for file_path in file_paths:
        tf = ROOT.TFile.Open(file_path)
        if not tf or tf.IsZombie():
            print(f"[WARNING] Cannot open {file_path} for {hname}", flush=True)
            continue

        tree = tf.Get(base_tree)
        nominal_required = [var_to_cut, var_to_plot, nominal_weight]
        for branch_name in extra_required_branches or []:
            if branch_name not in nominal_required:
                nominal_required.append(branch_name)
        if not tree or not tree_has_branches(tree, nominal_required):
            print(
                f"[WARNING] Skip unusable nominal input for {hname}: {file_path}",
                flush=True,
            )
            tf.Close()
            continue

        has_variation = bool(tree.GetBranch(branch))
        tf.Close()

        if has_variation:
            weight_expr = f"({nominal_weight})*({branch})"
        else:
            weight_expr = nominal_weight
            print(
                f"[WARNING] {branch} is missing in {file_path}; use nominal weight "
                f"for this component of {hname}.",
                flush=True,
            )

        component_name = f"{hname}_{uuid.uuid4().hex}"
        component = make_histogram(
            [file_path],
            base_tree,
            var_to_cut,
            var_to_plot,
            component_name,
            bin_edges,
            weight_expr,
            SR_or_SB,
            scale=scale,
            extra_cut=extra_cut,
            extra_required_branches=extra_required_branches,
        )
        total.Add(component)
        del component
        n_usable += 1

    if n_usable == 0:
        print(f"[WARNING] No usable inputs for {hname}; skip.", flush=True)
        return None

    print(f"[INFO] {hname}: combined yield = {total.Integral():.6g}", flush=True)
    return total


def process_era(era, args):
    region = args.region.upper()
    path_ = BASE_PATHS[era]
    base_tree = args.tree or TREE_NAME_BY_ERA[era]
    var_to_cut = MASS_BRANCH_BY_ERA[era]
    var_to_plot = args.score
    scale_mc = args.mc_scale if args.mc_scale is not None else MC_SCALE_BY_ERA[era]
    categories = categories_for_era(era)

    print("\n" + "=" * 100, flush=True)
    print("[INFO] Year:", era, flush=True)
    print("[INFO] Input path:", path_, flush=True)
    print("[INFO] Region:", region, flush=True)
    print("[INFO] Base tree:", base_tree, flush=True)
    print("[INFO] Mass branch:", var_to_cut, flush=True)
    print("[INFO] Score branch:", var_to_plot, flush=True)
    print("[INFO] MC scale:", scale_mc, flush=True)
    print("=" * 100, flush=True)

    weight_systs_common = [
        ("iso_MuonEff", "iso_MuonEffup", "iso_MuonEffdown"),
        ("id_MuonEff", "id_MuonEffup", "id_MuonEffdown"),
    ]

    weight_systs_by_process = {
        "ggH_hmm": [
            ("pdf_ggh", "PDF_uncertainty_up", "PDF_uncertainty_down"),
        ],
        "qqH_hmm": [
            ("pdf_qqh", "PDF_uncertainty_up", "PDF_uncertainty_down"),
            ("QCDscale_qqh", "qcd_unc_up", "qcd_unc_down"),
        ],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    outname = os.path.join(
        args.output_dir,
        f"vbf_ch3_vbfHmm_{var_to_plot}_{region}_{era}_domain_withSyst_splitDY.root",
    )
    fout = ROOT.TFile(outname, "RECREATE")
    if not fout or fout.IsZombie():
        raise RuntimeError(f"Could not create output file: {outname}")

    try:
        for category, files in categories.items():
            extra_cut = DY_SPLIT_SELECTIONS.get(category)
            extra_required_branches = (
                ["n_jets_matched_genjet"] if extra_cut else []
            )
            if extra_cut:
                print(f"[INFO] {category} event selection: {extra_cut}", flush=True)
            full_paths = [os.path.join(path_, filename) for filename in files]
            existing_paths = []
            for file_path in full_paths:
                if os.path.exists(file_path):
                    existing_paths.append(file_path)
                else:
                    print(f"[WARNING] Missing input file: {file_path}", flush=True)

            if not existing_paths:
                print(f"[WARNING] No files available for {era} {category}; skip.", flush=True)
                continue

            is_data = category == "data_obs"
            nominal_weight = "1.0" if is_data else "eventWeight"
            scale = 1.0 if is_data else scale_mc

            if is_data:
                tree_list = [base_tree]
            else:
                tree_list = discover_trees_for_files(existing_paths, base_tree)
                if not tree_list:
                    print(f"[WARNING] No matching trees found for {era} {category}; skip.", flush=True)
                    continue

            for tree_name in tqdm(tree_list, desc=f"{era} {category}"):
                suffix = syst_suffix_from_tree(tree_name, base_tree)
                hist_name = category if suffix == "" else f"{category}_{suffix}"

                h = make_histogram(
                    existing_paths,
                    tree_name,
                    var_to_cut,
                    var_to_plot,
                    hist_name,
                    BIN_EDGES,
                    nominal_weight,
                    region,
                    scale=scale,
                    fallback_tree_name=base_tree if suffix else None,
                    extra_cut=extra_cut,
                    extra_required_branches=extra_required_branches,
                )
                write_hist(fout, h, hist_name)
                del h

                if is_data or suffix != "":
                    continue

                for syst_name, up_branch, down_branch in weight_systs_common:
                    h_up = build_weight_syst_hist(
                        existing_paths,
                        base_tree,
                        category,
                        syst_name,
                        up_branch,
                        "Up",
                        var_to_cut,
                        var_to_plot,
                        BIN_EDGES,
                        nominal_weight,
                        region,
                        scale,
                        extra_cut=extra_cut,
                        extra_required_branches=extra_required_branches,
                    )
                    write_hist(fout, h_up, f"{category}_{syst_name}Up")
                    if h_up is not None:
                        del h_up

                    h_down = build_weight_syst_hist(
                        existing_paths,
                        base_tree,
                        category,
                        syst_name,
                        down_branch,
                        "Down",
                        var_to_cut,
                        var_to_plot,
                        BIN_EDGES,
                        nominal_weight,
                        region,
                        scale,
                        extra_cut=extra_cut,
                        extra_required_branches=extra_required_branches,
                    )
                    write_hist(fout, h_down, f"{category}_{syst_name}Down")
                    if h_down is not None:
                        del h_down

                for syst_name, up_branch, down_branch in weight_systs_by_process.get(category, []):
                    h_up = build_weight_syst_hist(
                        existing_paths,
                        base_tree,
                        category,
                        syst_name,
                        up_branch,
                        "Up",
                        var_to_cut,
                        var_to_plot,
                        BIN_EDGES,
                        nominal_weight,
                        region,
                        scale,
                        extra_cut=extra_cut,
                        extra_required_branches=extra_required_branches,
                    )
                    write_hist(fout, h_up, f"{category}_{syst_name}Up")
                    if h_up is not None:
                        del h_up

                    h_down = build_weight_syst_hist(
                        existing_paths,
                        base_tree,
                        category,
                        syst_name,
                        down_branch,
                        "Down",
                        var_to_cut,
                        var_to_plot,
                        BIN_EDGES,
                        nominal_weight,
                        region,
                        scale,
                        extra_cut=extra_cut,
                        extra_required_branches=extra_required_branches,
                    )
                    write_hist(fout, h_down, f"{category}_{syst_name}Down")
                    if h_down is not None:
                        del h_down
    finally:
        fout.Close()

    print("[INFO] Wrote:", outname, flush=True)
    return outname


def main():
    args = get_args()
    selected_eras = ALL_ERAS if args.year == "all" else [args.year]
    outputs = []

    for era in selected_eras:
        outputs.append(process_era(era, args))

    print("\n[INFO] Finished. Output files:", flush=True)
    for output in outputs:
        print("  ", output, flush=True)


if __name__ == "__main__":
    main()
