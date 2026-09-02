#!/usr/bin/env python3
"""Add EWK parton-shower Up/Down shapes to existing Combine ROOT files."""
# python3 scripts/add_ewk_parton_shower_shapes.py   --shape-dir noBr4_simple_dnn_DAandNoDomain_20260721_allYears_gpu_v1_Nor/ --no-shape-only
# python3 scripts/add_ewk_parton_shower_shapes.py   --shape-dir noBr4_simple_dnn_DAandNoDomain_20260721_allYears_gpu_v1_mergedNor/ --no-shape-only  --shape-file-suffix "" 

from __future__ import annotations

import argparse
import array
import math
import os
import uuid
from pathlib import Path

import ROOT


YEARS = ("2022", "2022EE", "2023", "2023BPix", "2024", "2025")
REGIONS = ("SR", "SB")
BASE_PATHS = {
    "2022": "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2022_ggHVBF/simpleDNN_DA_0722_2022/",
    "2022EE": "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2022EE_ggHVBF/simpleDNN_DA_0722_2022EE/",
    "2023": "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2023_ggHVBF/simpleDNN_DA_0722_2023/",
    "2023BPix": "/eos/user/z/zhangxu/sharing/hmm/skimmed_ntuples/2023BPix_ggHVBF/simpleDNN_DA_0722_2023BPix/",
    "2024": "/eos/user/h/hakou/Hmumu_Share/qguo/2024_v5_SRSB/skimmed_ntuples/SRSB_Nor/simpleDNN_DA_0722_2024/",
    "2025": "/eos/user/h/hakou/Hmumu_Share/qguo/2025_v5_SRSB_v2/skimmed_ntuples/SRSB_Nor/simpleDNN_DA_0722_2025/",
}
MASS_BRANCH = {
    "2022": "diMufsr_rc_BSC_mass",
    "2022EE": "diMufsr_rc_BSC_mass",
    "2023": "diMufsr_rc_BSC_mass",
    "2023BPix": "diMufsr_rc_BSC_mass",
    "2024": "diMufsr_kit_BSC_mass",
    "2025": "diMufsr_kit_BSC_mass",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape-dir",
        type=Path,
        required=True,
        help="Directory containing the already-produced Combine shape ROOT files.",
    )
    parser.add_argument("--year", choices=("all",) + YEARS, default="all")
    parser.add_argument("--region", choices=("all",) + REGIONS, default="all")
    parser.add_argument("--score", default="dnn_t")
    parser.add_argument("--tree", default="data_two_jet_m110To150_VBF")
    parser.add_argument("--weight", default="eventWeight")
    parser.add_argument("--mc-scale", type=float, default=1.0)
    parser.add_argument("--nuisance", default="EWKPS")
    parser.add_argument(
        "--alternative-file",
        default="EWK_2Mu2J_105to160_pythia_idiso.root",
    )
    parser.add_argument(
        "--shape-file-suffix",
        default="_Nor",
        help="Suffix before .root in the target Combine shape filename.",
    )
    parser.add_argument(
        "--no-shape-only",
        action="store_true",
        help="Keep the alternative sample's normalization instead of matching nominal EWKZ.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Override BASE_PATHS; only valid when one year is selected.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selection(mass: str, region: str) -> str:
    if region == "SR":
        return f"({mass} >= 115 && {mass} <= 135)"
    return f"(({mass} >= 110 && {mass} < 115) || ({mass} > 135 && {mass} <= 150))"


def shape_filename(score: str, region: str, year: str, suffix: str) -> str:
    return (
        f"vbf_ch3_vbfHmm_{score}_{region}_{year}_"
        f"domain_withSyst_splitDY{suffix}.root"
    )


def bin_edges(hist: ROOT.TH1) -> list[float]:
    axis = hist.GetXaxis()
    return [axis.GetBinLowEdge(i) for i in range(1, hist.GetNbinsX() + 1)] + [
        axis.GetBinUpEdge(hist.GetNbinsX())
    ]


def build_alternative(
    input_path: Path,
    tree_name: str,
    mass_branch: str,
    score_branch: str,
    weight: str,
    region: str,
    edges: list[float],
    scale: float,
) -> ROOT.TH1D:
    source = ROOT.TFile.Open(str(input_path), "READ")
    if not source or source.IsZombie():
        raise RuntimeError(f"Cannot open alternative EWK file: {input_path}")
    tree = source.Get(tree_name)
    if not tree or not tree.InheritsFrom("TTree"):
        source.Close()
        raise RuntimeError(f"Missing tree {tree_name!r} in {input_path}")
    required = (mass_branch, score_branch, weight)
    missing = [name for name in required if not tree.GetBranch(name)]
    if missing:
        source.Close()
        raise RuntimeError(f"Missing branches in {input_path}: {', '.join(missing)}")

    ROOT.gROOT.cd()
    temporary_name = "tmp_ewk_ps_" + uuid.uuid4().hex
    hist = ROOT.TH1D(temporary_name, "", len(edges) - 1, array.array("d", edges))
    hist.Sumw2()
    draw_weight = f"({weight})*({scale})*({selection(mass_branch, region)})"
    tree.Draw(f"{score_branch} >> {temporary_name}", draw_weight, "goff")
    hist.SetDirectory(0)
    source.Close()
    return hist


def reflected_down(nominal: ROOT.TH1, up: ROOT.TH1, name: str) -> tuple[ROOT.TH1, int]:
    down = nominal.Clone(name)
    down.SetDirectory(0)
    clipped = 0
    for bin_index in range(0, nominal.GetNbinsX() + 2):
        value = 2.0 * nominal.GetBinContent(bin_index) - up.GetBinContent(bin_index)
        if value < 0:
            value = 0.0
            clipped += 1
        down.SetBinContent(bin_index, value)
        down.SetBinError(
            bin_index,
            math.sqrt(
                4.0 * nominal.GetBinError(bin_index) ** 2
                + up.GetBinError(bin_index) ** 2
            ),
        )
    return down, clipped


def write_shapes(path: Path, year: str, region: str, args: argparse.Namespace) -> None:
    mode = "READ" if args.dry_run else "UPDATE"
    output = ROOT.TFile.Open(str(path), mode)
    if not output or output.IsZombie():
        raise RuntimeError(f"Cannot open target shape file: {path}")
    nominal_source = output.Get("EWKZ")
    if not nominal_source or not nominal_source.InheritsFrom("TH1"):
        output.Close()
        raise RuntimeError(f"Missing nominal EWKZ histogram in {path}")
    nominal = nominal_source.Clone(f"nominal_EWKZ_{year}_{region}")
    nominal.SetDirectory(0)
    edges = bin_edges(nominal)

    input_dir = args.input_dir if args.input_dir else Path(BASE_PATHS[year])
    alternative_path = input_dir / args.alternative_file
    up = build_alternative(
        alternative_path,
        args.tree,
        MASS_BRANCH[year],
        args.score,
        args.weight,
        region,
        edges,
        args.mc_scale,
    )
    raw_integral = up.Integral()
    nominal_integral = nominal.Integral()
    if not args.no_shape_only:
        if raw_integral <= 0 or nominal_integral <= 0:
            output.Close()
            raise RuntimeError(
                f"Cannot shape-normalize nonpositive yields: nominal={nominal_integral}, "
                f"alternative={raw_integral}"
            )
        up.Scale(nominal_integral / raw_integral)

    up_name = f"EWKZ_{args.nuisance}Up"
    down_name = f"EWKZ_{args.nuisance}Down"
    up.SetName(up_name)
    up.SetTitle(up_name)
    down, clipped = reflected_down(nominal, up, down_name)
    down.SetTitle(down_name)

    print(f"\n{year} {region}: {path}")
    print(f"  nominal integral       = {nominal_integral:.9g}")
    print(f"  alternative raw        = {raw_integral:.9g}")
    print(f"  Up integral            = {up.Integral():.9g}")
    print(f"  reflected Down integral= {down.Integral():.9g}")
    print(f"  clipped negative bins  = {clipped}")
    if args.dry_run:
        output.Close()
        return

    output.cd()
    up.Write(up_name, ROOT.TObject.kOverwrite)
    down.Write(down_name, ROOT.TObject.kOverwrite)
    output.Write()
    output.Close()

    check = ROOT.TFile.Open(str(path), "READ")
    if not check.Get(up_name) or not check.Get(down_name):
        check.Close()
        raise RuntimeError(f"Failed to verify {up_name}/{down_name} in {path}")
    check.Close()
    print(f"  wrote {up_name} and {down_name}")


def main() -> None:
    args = parse_args()
    if args.input_dir and args.year == "all":
        raise SystemExit("--input-dir requires a single --year")
    ROOT.gROOT.SetBatch(True)
    years = YEARS if args.year == "all" else (args.year,)
    regions = REGIONS if args.region == "all" else (args.region,)
    for year in years:
        for region in regions:
            path = args.shape_dir / shape_filename(
                args.score, region, year, args.shape_file_suffix
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            write_shapes(path, year, region, args)


if __name__ == "__main__":
    main()
