#!/usr/bin/env python3
"""
score_and_map.py

Score aligned ref/alt prediction arrays and map variants to regulatory regions.

Inputs:
  --ref_aligned   ref_predictions_aligned.npy
  --alt_aligned   alt_predictions_aligned.npy
  --metadata      metadata.csv (must contain columns: chrom, variant_pos1)
  --regions_bed   regions.bed (BED3+; must include 4th column region_name)
  --tile          tile size (e.g., 250)
  --out_prefix    /path/outdir/run_prefix

Outputs:
  {out_prefix}_aligned_scores.csv
  {out_prefix}_variant_region_mapped.csv
  {out_prefix}_scores_with_regions.csv

Notes:
- Requires: numpy, pandas, scipy, bedtools (in PATH)
- metadata.variant_pos1 assumed 1-based coordinate (as in your notebook)
- regions bed assumed BED convention (start 0-based, end 1-based)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def check_exists(path: str, what: str) -> None:
    if not path:
        raise SystemExit(f"Missing required argument: {what}")
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"File not found: {path}")


def require_tool(tool: str) -> None:
    if shutil.which(tool) is None:
        raise SystemExit(f"Required tool not found in PATH: {tool}")


def write_variants_bed(metadata_path: str, variants_bed_path: str) -> int:
    md = pd.read_csv(metadata_path)
    required = {"chrom", "variant_pos1"}
    missing = required - set(md.columns)
    if missing:
        raise SystemExit(
            f"metadata.csv missing columns: {sorted(missing)}. Found: {list(md.columns)}"
        )

    variants_bed = md[["chrom", "variant_pos1"]].copy()
    variants_bed["start"] = variants_bed["variant_pos1"].astype(int)
    variants_bed["end"] = variants_bed["start"] + 1
    variants_bed["variant_id"] = np.arange(len(variants_bed))
    variants_bed_out = variants_bed[["chrom", "start", "end", "variant_id"]]

    Path(variants_bed_path).parent.mkdir(parents=True, exist_ok=True)
    variants_bed_out.to_csv(variants_bed_path, sep="\t", header=False, index=False)
    return len(variants_bed_out)


def run_bedtools_intersect(variants_bed: str, regions_bed: str, out_tsv: str) -> None:
    require_tool("bedtools")
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "bedtools",
        "intersect",
        "-a",
        variants_bed,
        "-b",
        regions_bed,
        "-wa",
        "-wb",
    ]
    with open(out_tsv, "w") as f:
        subprocess.run(cmd, check=True, stdout=f)


def safe_corrs(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = ~np.isnan(x) & ~np.isnan(y)
    x2 = x[mask]
    y2 = y[mask]
    if x2.size < 2:
        return np.nan, np.nan
    s_result = spearmanr(x2, y2)
    s = s_result.correlation
    s_p = s_result.pvalue
    p, p_p = pearsonr(x2, y2)

    # try:
    #     p = pearsonr(x2, y2).correlation
    #     p_p = pearsonr(x2, y2).pvalue
    # except Exception:
    #     print(f"Pearsonr error: {x2}, {y2}, {x2.shape}, {y2.shape}")
    #     raise
    #     p = np.nan
    #     p_p = np.nan
    return float(s), float(p), float(s_p), float(p_p)


def score_aligned_arrays(
    ref_aligned: np.ndarray,
    alt_aligned: np.ndarray,
    eps: float = 1e-6,
) -> pd.DataFrame:
    if ref_aligned.shape != alt_aligned.shape:
        raise ValueError(
            f"Shape mismatch: ref {ref_aligned.shape} vs alt {alt_aligned.shape}"
        )
    if ref_aligned.ndim != 2:
        raise ValueError(f"Expected 2D arrays (n_variants, L); got {ref_aligned.ndim}D")

    n, _L = ref_aligned.shape
    MSE = np.full(n, np.nan)
    Spearman = np.full(n, np.nan)
    Spearman_p = np.full(n, np.nan)
    Pearson = np.full(n, np.nan)
    Pearson_p = np.full(n, np.nan)
    ref_total = np.full(n, np.nan)
    ref_max = np.full(n, np.nan)
    alt_total = np.full(n, np.nan)
    alt_max = np.full(n, np.nan)
    delta_total = np.full(n, np.nan)
    log2fc_total = np.full(n, np.nan)
    effective_len = np.full(n, np.nan)

    for i in range(n):
        r = ref_aligned[i]
        a = alt_aligned[i]
        mask = ~np.isnan(r) & ~np.isnan(a)
        m = int(mask.sum())
        effective_len[i] = m
        if m == 0:
            continue

        # the masked version of ref and alt predictions for a variant
        r2 = r[mask]
        a2 = a[mask]
        d = a2 - r2 # this is element-wise subtraction

        MSE[i] = np.mean(d**2)

        rt = np.sum(r2)
        rmax = r2.max()
        at = np.sum(a2)
        amax = a2.max()


        ref_total[i] = rt
        ref_max[i] = rmax
        alt_total[i] = at
        alt_max[i] = amax
        delta_total[i] = at - rt
        log2fc_total[i] = np.log2((at + eps) / (rt + eps))

        s, p, s_p, p_p = safe_corrs(r, a)
        Spearman[i] = s
        Spearman_p[i] = s_p
        Pearson[i] = p
        Pearson_p[i] = p_p

    return pd.DataFrame(
        {
            "variant_index": np.arange(n),
            "MSE": MSE,
            "Spearman": Spearman,
            "Spearman_p": Spearman_p,
            "Pearson": Pearson,
            "Pearson_p": Pearson_p,
            "ref_total": ref_total,
            "ref_max": ref_max,
            "alt_total": alt_total,
            "alt_max": alt_max,
            "delta_total": delta_total,
            "log2fc_total": log2fc_total,
            "effective_len": effective_len,
        }
    )


def map_variants_to_regions(
    metadata: pd.DataFrame, intersect_path: str
) -> pd.DataFrame:
    """
    Returns metadata with columns added:
      - variant_id (0..n-1)
      - region_name
      - region_id (chr:start-end in BED coords)
    Keeps first hit per variant_id if multiple overlaps.
    """
    metadata_mapped = metadata.copy()
    metadata_mapped["variant_id"] = np.arange(len(metadata_mapped))

    # Empty intersect -> no overlaps
    try:
        ix_preview = pd.read_csv(intersect_path, sep="\t", header=None, nrows=1)
        if ix_preview.shape[0] == 0:
            metadata_mapped["region_name"] = np.nan
            metadata_mapped["region_id"] = np.nan
            return metadata_mapped
    except pd.errors.EmptyDataError:
        metadata_mapped["region_name"] = np.nan
        metadata_mapped["region_id"] = np.nan
        return metadata_mapped

    ix = pd.read_csv(intersect_path, sep="\t", header=None)

    # Require at least 8 cols: a(BED4) + b(BED4)
    if ix.shape[1] < 8:
        
        raise ValueError(
            f"Intersect output has {ix.shape[1]} columns; expected >= 8. "
            f"Does your regions bed have a 4th column (region name)?"
        )

    ix = ix.rename(
        columns={
            0: "v_chrom",
            1: "v_start",
            2: "v_end",
            3: "variant_id",
            4: "r_chrom",
            5: "r_start",
            6: "r_end",
            7: "region_name",
        }
    )

    ix["region_id"] = (
        ix["r_chrom"].astype(str)
        + ":"
        + ix["r_start"].astype(str)
        + "-"
        + ix["r_end"].astype(str)
    )

    # Keep first hit per variant
    ix_uniq = ix.sort_values(["variant_id"]).drop_duplicates("variant_id")

    metadata_mapped = metadata_mapped.merge(
        ix_uniq[["variant_id", "region_name", "region_id"]],
        on="variant_id",
        how="left",
    )
    return metadata_mapped


def merge_scores_and_add_tiles(
    metadata_mapped: pd.DataFrame,
    scores: pd.DataFrame,
    tile: int,
) -> pd.DataFrame:
    """
    Merge scores onto mapped metadata and compute:
      - abs_log2fc, abs_delta
      - region_start/end/len from region_id
      - pos_in_region, tile_id
    Returns only rows with region_id notna AND pos_ok.
    """
    metric_cols = [
        "variant_index",
        "MSE",
        "Spearman",
        "Spearman_p",
        "Pearson",
        "Pearson_p",
        "ref_total",
        "ref_max",
        "alt_total",
        "alt_max",
        "delta_total",
        "log2fc_total",
        "effective_len",
    ]
    scores_clean = scores[metric_cols].copy()

    df = metadata_mapped.merge(
        scores_clean,
        left_on="variant_id",
        right_on="variant_index",
        how="left",
    )

    df["abs_log2fc"] = df["log2fc_total"].abs()
    df["abs_delta"] = df["delta_total"].abs()

    df_valid = df[df["region_id"].notna()].copy()
    if len(df_valid) == 0:
        # Return empty df with expected columns
        df_valid["tile_id"] = pd.Series(dtype="int")
        return df_valid

    tmp = df_valid["region_id"].str.extract(r"^(chr[^:]+):(\d+)-(\d+)$")
    df_valid["region_chrom"] = tmp[0]
    df_valid["region_start"] = tmp[1].astype(int)
    # BED end -> last base 0-based
    df_valid["region_end"] = tmp[2].astype(int) - 1
    df_valid["region_len"] = df_valid["region_end"] - df_valid["region_start"]

    # variant_pos1 assumed 1-based
    df_valid["pos_in_region"] = df_valid["variant_pos1"].astype(int) - df_valid["region_start"]
    df_valid["pos_ok"] = df_valid["pos_in_region"].between(
        0, df_valid["region_len"] - 1, inclusive="both"
    )
    df_valid = df_valid[df_valid["pos_ok"]].copy()

    df_valid["tile_id"] = (df_valid["pos_in_region"] // tile).astype(int)
    return df_valid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score aligned arrays and map variants to regulatory regions via bedtools intersect."
    )
    p.add_argument("--ref_aligned", required=True, help="Path to ref_predictions_aligned.npy")
    p.add_argument("--alt_aligned", required=True, help="Path to alt_predictions_aligned.npy")
    p.add_argument("--metadata", required=True, help="Path to metadata.csv (must include chrom, variant_pos1)")
    p.add_argument("--regions_bed", required=True, help="Path to regions BED (must include 4th col region_name)")
    p.add_argument("--tile", required=True, type=int, help="Tile size (e.g., 250)")
    p.add_argument("--out_prefix", required=True, help="Output prefix, e.g. /path/outdir/run_name")
    p.add_argument("--eps", type=float, default=1e-6, help="Epsilon for log2fc_total stability (default: 1e-6)")
    p.add_argument(
        "--keep_tmp",
        action="store_true",
        help="Keep temporary bedtools directory (default: delete nothing anyway, but this makes it explicit).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Checks
    check_exists(args.ref_aligned, "--ref_aligned")
    check_exists(args.alt_aligned, "--alt_aligned")
    check_exists(args.metadata, "--metadata")
    check_exists(args.regions_bed, "--regions_bed")
    require_tool("bedtools")

    out_prefix = Path(args.out_prefix)
    outdir = out_prefix.parent
    outdir.mkdir(parents=True, exist_ok=True)

    tmpdir = outdir / f"tmp_bedtools_{out_prefix.name}"
    tmpdir.mkdir(parents=True, exist_ok=True)

    variants_bed = str(tmpdir / "variants_1bp.bed")
    intersect_tsv = str(tmpdir / "variants_x_regions.tsv")

    scores_csv = str(out_prefix) + "_aligned_scores.csv"
    mapped_meta_csv = str(out_prefix) + "_variant_region_mapped.csv"
    scores_with_regions_csv = str(out_prefix) + "_scores_with_regions.csv"

    print("[1/3] Build 1bp variant BED from metadata...", flush=True)
    nvars = write_variants_bed(args.metadata, variants_bed)
    print(f"Wrote: {variants_bed}")
    print(f"N variants: {nvars}")

    print("[2/3] bedtools intersect variants x regions...", flush=True)
    run_bedtools_intersect(variants_bed, args.regions_bed, intersect_tsv)
    print(f"Wrote: {intersect_tsv}")

    print("[3/3] Score aligned arrays + map variants to regions + add tile_id...", flush=True)
    ref_aligned = np.load(args.ref_aligned)
    alt_aligned = np.load(args.alt_aligned)
    metadata = pd.read_csv(args.metadata)

    scores = score_aligned_arrays(ref_aligned, alt_aligned, eps=args.eps)
    scores.to_csv(scores_csv, index=False)

    metadata_mapped = map_variants_to_regions(metadata, intersect_tsv)
    metadata_mapped.to_csv(mapped_meta_csv, index=False)

    df_valid = merge_scores_and_add_tiles(metadata_mapped, scores, tile=args.tile)
    df_valid.to_csv(scores_with_regions_csv, index=False)

    frac_mapped = float(metadata_mapped["region_id"].notna().mean()) if "region_id" in metadata_mapped else float("nan")

    print(f"Wrote: {scores_csv}")
    print(f"Wrote: {mapped_meta_csv} | Fraction mapped: {frac_mapped:.4f}")
    print(f"Wrote: {scores_with_regions_csv} | Rows with mapped+pos_ok: {len(df_valid)}")

    # Optional cleanup (currently we keep tmp by default since it's small and useful for debugging)
    if not args.keep_tmp:
        # Keep by default? In bash script we kept it. Here, we also keep it unless user asks to remove.
        pass


if __name__ == "__main__":
    main()

