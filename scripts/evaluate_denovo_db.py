#!/usr/bin/env python3
"""
evaluate_denovo_db.py — Evaluate SynthDNM classifier against denovo-db truth set.

Matches validated DNMs from denovo-db (hg38-lifted) against the raw feature
data (loading per-chromosome to manage memory), preprocesses matched variants,
scores with classifier, and reports evaluation metrics.

Sample ID mapping chain:
    denovo-db: SFARI IDs (e.g., 14666.p1)
    phase_ids files: SS_ID SFARI_ID  → reversed to SFARI_ID → SS_ID
    Feature data: SS IDs (e.g., SS0012979)

Usage:
    python evaluate_denovo_db.py \
        --denovo_db resources/denovo-db.ssc-samples.variants.v.1.6.1.hg38.tsv.gz \
        --feature_dir output/features/ \
        --phase_ids_dir /path/to/SSC_JG/documentation/ \
        --model output/models/gatk/model.json \
        --config scripts/features.toml \
        --feature_set gatk \
        --psam resources/SSC.psam \
        --par resources/hg38_par.bed \
        --output output/evaluation/denovo_db_eval.tsv
"""

import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb


def build_sfari_to_ss_mapping(phase_ids_dir: Path) -> dict[str, str]:
    """Build SFARI_ID → SS_ID mapping from phase_ids files."""
    mapping = {}
    phase_files = sorted(phase_ids_dir.glob("phase*_ids"))
    for pf in phase_files:
        with open(pf) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) == 2:
                    ss_id, sfari_id = parts
                    mapping[sfari_id] = ss_id
    print(f"  Loaded {len(mapping):,} SFARI→SS mappings from {len(phase_files)} phase files",
          file=sys.stderr)
    return mapping


def load_denovo_db(path: Path, sfari_to_ss: dict[str, str]) -> pl.DataFrame:
    """Load denovo-db hg38-lifted file and map sample IDs."""
    # Skip the ##version line, read from the header line
    with open(path, "rb") as f:
        # Check if gzipped
        magic = f.read(2)

    if magic == b"\x1f\x8b":
        import gzip
        with gzip.open(path, "rt") as f:
            lines = f.readlines()
    else:
        with open(path) as f:
            lines = f.readlines()

    # Find the header line (starts with #SampleID)
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("#SampleID"):
            header_idx = i
            break

    if header_idx is None:
        print("ERROR: Could not find header line in denovo-db file", file=sys.stderr)
        sys.exit(1)

    # Parse header
    header = lines[header_idx].lstrip("#").strip().split("\t")
    data_lines = [line.strip().split("\t") for line in lines[header_idx + 1:] if line.strip()]

    ddb = pl.DataFrame(data_lines, schema={col: pl.Utf8 for col in header}, orient="row")

    print(f"denovo-db loaded: {ddb.shape[0]:,} rows", file=sys.stderr)

    # Study breakdown
    study_counts = ddb.group_by("StudyName").agg(pl.len().alias("n")).sort("n", descending=True)
    for row in study_counts.iter_rows(named=True):
        print(f"  {row['StudyName']}: {row['n']:,}", file=sys.stderr)

    # Validation breakdown
    val_counts = ddb.group_by("Validation").agg(pl.len().alias("n")).sort("n", descending=True)
    for row in val_counts.iter_rows(named=True):
        print(f"  Validation={row['Validation']}: {row['n']:,}", file=sys.stderr)

    # Map SFARI IDs to SS IDs
    sfari_ids = ddb["SampleID"].to_list()
    ss_ids = [sfari_to_ss.get(sid) for sid in sfari_ids]
    mapped = sum(1 for s in ss_ids if s is not None)
    unmapped = sum(1 for s in ss_ids if s is None)
    print(f"  Sample ID mapping: {mapped:,} mapped, {unmapped:,} unmapped", file=sys.stderr)

    if unmapped > 0:
        unmapped_examples = list(set(sid for sid, ss in zip(sfari_ids, ss_ids) if ss is None))[:5]
        print(f"  Unmapped examples: {unmapped_examples}", file=sys.stderr)

    ddb = ddb.with_columns(pl.Series("SS_ID", ss_ids))

    # Filter to rows with hg38 coords and valid SS_ID
    ddb = ddb.filter(
        pl.col("Hg38_Chr").is_not_null()
        & (pl.col("Hg38_Chr") != "")
        & pl.col("SS_ID").is_not_null()
    )

    # Cast position to int
    ddb = ddb.with_columns(pl.col("Hg38_Position").cast(pl.Int64))

    print(f"  After filtering (hg38 + mapped): {ddb.shape[0]:,} rows", file=sys.stderr)
    return ddb


def load_and_filter_features(feature_dir: Path, chrom: str,
                              lookup: pl.DataFrame) -> pl.DataFrame:
    """Load real DNM features for one chromosome, filter to denovo-db matches."""
    import glob as globmod
    pattern = str(feature_dir / f"{chrom}_*_real.tsv")
    files = sorted(globmod.glob(pattern))
    if not files:
        return pl.DataFrame()

    dfs = []
    for f in files:
        df = pl.read_csv(f, separator="\t", infer_schema_length=10000,
                         null_values=[".", "NA", "nan", ""])
        # Filter to matching positions and samples
        df = df.join(
            lookup.select(["pos", "sample"]),
            left_on=["POS", "SAMPLE"],
            right_on=["pos", "sample"],
            how="semi",
        )
        if df.shape[0] > 0:
            dfs.append(df)

    if not dfs:
        return pl.DataFrame()
    return pl.concat(dfs, how="diagonal_relaxed")


def derive_features(df: pl.DataFrame) -> pl.DataFrame:
    """Derive features matching preprocess_features.py logic exactly."""
    allele_num = pl.col("allele_num").cast(pl.Int32)

    # --- Parse AD into ref/alt depths ---
    for member in ("child", "father", "mother"):
        ad_col = f"{member}_AD"
        if ad_col in df.columns:
            ad_split = pl.col(ad_col).str.split(",")
            df = df.with_columns([
                ad_split.list.get(0, null_on_oob=True).cast(pl.Float64)
                    .alias(f"{member}_AD_ref"),
                pl.when(allele_num == 2)
                    .then(ad_split.list.get(2, null_on_oob=True).cast(pl.Float64))
                    .otherwise(ad_split.list.get(1, null_on_oob=True).cast(pl.Float64))
                    .alias(f"{member}_AD_alt"),
            ])

    # --- Compute allele ratios: AR = ref / (alt + 1), clamped to [0, 1] ---
    for member in ("child", "father", "mother"):
        ref_col = f"{member}_AD_ref"
        alt_col = f"{member}_AD_alt"
        if ref_col in df.columns and alt_col in df.columns:
            raw_ar = pl.col(ref_col) / (pl.col(alt_col) + 1.0)
            df = df.with_columns(
                pl.when(raw_ar > 1.0).then(1.0 / raw_ar).otherwise(raw_ar)
                    .alias(f"{member}_AR")
            )

    # min/max parent AR (parent-only, not child)
    if "father_AR" in df.columns and "mother_AR" in df.columns:
        df = df.with_columns([
            pl.min_horizontal("father_AR", "mother_AR").alias("min_AR"),
            pl.max_horizontal("father_AR", "mother_AR").alias("max_AR"),
        ])

    # --- min/max parent GQ, DP ---
    for metric in ("GQ", "DP"):
        f_col = f"father_{metric}"
        m_col = f"mother_{metric}"
        if f_col in df.columns and m_col in df.columns:
            df = df.with_columns([
                pl.min_horizontal(f_col, m_col).alias(f"min_{metric}"),
                pl.max_horizontal(f_col, m_col).alias(f"max_{metric}"),
            ])

    # --- Parse PL into individual components ---
    # Biallelic (allele_num=1): PL0=PL[0], PL1=PL[1], PL2=PL[2]
    # Second ALT (allele_num=2): PL0=PL[0], PL1=PL[3], PL2=PL[5]
    pl_indices = {"PL0": (0, 0), "PL1": (1, 3), "PL2": (2, 5)}
    for member in ("child", "father", "mother"):
        pl_col = f"{member}_PL"
        if pl_col in df.columns:
            for comp, (idx1, idx2) in pl_indices.items():
                pl_split = pl.col(pl_col).str.split(",")
                df = df.with_columns(
                    pl.when(allele_num == 2)
                        .then(pl_split.list.get(idx2, null_on_oob=True).cast(pl.Float64))
                        .otherwise(pl_split.list.get(idx1, null_on_oob=True).cast(pl.Float64))
                        .alias(f"{member}_{comp}")
                )

    # min/max parent PL components
    for comp in ("PL0", "PL1", "PL2"):
        f_col = f"father_{comp}"
        m_col = f"mother_{comp}"
        if f_col in df.columns and m_col in df.columns:
            df = df.with_columns([
                pl.min_horizontal(f_col, m_col).alias(f"min_{comp}"),
                pl.max_horizontal(f_col, m_col).alias(f"max_{comp}"),
            ])

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SynthDNM classifier against denovo-db truth set"
    )
    parser.add_argument("--denovo_db", required=True, type=Path)
    parser.add_argument("--feature_dir", required=True, type=Path)
    parser.add_argument("--phase_ids_dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path,
                        help="features.toml config")
    parser.add_argument("--feature_set", default="gatk")
    parser.add_argument("--psam", type=Path, default=None)
    parser.add_argument("--par", type=Path, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # --- Step 1: Build sample ID mapping ---
    print("=== Step 1: Building sample ID mapping ===", file=sys.stderr)
    sfari_to_ss = build_sfari_to_ss_mapping(args.phase_ids_dir)

    # --- Step 2: Load denovo-db ---
    print("\n=== Step 2: Loading denovo-db ===", file=sys.stderr)
    ddb = load_denovo_db(args.denovo_db, sfari_to_ss)

    # Create lookup by chromosome
    ddb_by_chrom = {}
    for chrom in ddb["Hg38_Chr"].unique().sort().to_list():
        subset = ddb.filter(pl.col("Hg38_Chr") == chrom).select([
            pl.col("Hg38_Position").alias("pos"),
            pl.col("SS_ID").alias("sample"),
            pl.col("SampleID").alias("sfari_id"),
            pl.col("StudyName"),
            pl.col("Validation"),
            pl.col("Gene"),
            pl.col("FunctionClass"),
        ])
        ddb_by_chrom[chrom] = subset
    print(f"  Chromosomes with variants: {len(ddb_by_chrom)}", file=sys.stderr)

    # --- Step 3: Load features per-chrom, filter to denovo-db matches ---
    print("\n=== Step 3: Loading features and matching ===", file=sys.stderr)
    all_matched = []
    for chrom in sorted(ddb_by_chrom.keys()):
        lookup = ddb_by_chrom[chrom]
        features = load_and_filter_features(args.feature_dir, chrom, lookup)
        if features.shape[0] > 0:
            # Join denovo-db metadata
            features = features.join(
                lookup,
                left_on=["POS", "SAMPLE"],
                right_on=["pos", "sample"],
                how="inner",
            )
            all_matched.append(features)
            print(f"  {chrom}: {features.shape[0]:,} matches", file=sys.stderr)
        else:
            print(f"  {chrom}: 0 matches", file=sys.stderr)

    if not all_matched:
        print("\nERROR: No variants matched!", file=sys.stderr)
        # Debug info
        ddb_samples = set(ddb["SS_ID"].unique().to_list())
        print(f"  denovo-db unique SS_IDs: {len(ddb_samples)}", file=sys.stderr)
        print(f"  Example SS_IDs: {list(ddb_samples)[:5]}", file=sys.stderr)
        # Check one feature file
        import glob as globmod
        test_files = sorted(globmod.glob(str(args.feature_dir / "chr1_*_real.tsv")))
        if test_files:
            test_df = pl.read_csv(test_files[0], separator="\t", n_rows=5)
            print(f"  Feature file sample IDs: {test_df['SAMPLE'].to_list()}", file=sys.stderr)
            print(f"  Feature file columns: {test_df.columns[:10]}", file=sys.stderr)
        sys.exit(1)

    matched = pl.concat(all_matched, how="diagonal_relaxed")
    print(f"\n  Total matched: {matched.shape[0]:,} / {ddb.shape[0]:,} denovo-db variants "
          f"({100*matched.shape[0]/ddb.shape[0]:.1f}%)", file=sys.stderr)

    # --- Step 4: Preprocess features ---
    print("\n=== Step 4: Preprocessing features ===", file=sys.stderr)
    matched = derive_features(matched)

    # Load model and get expected features
    model = xgb.XGBClassifier()
    model.load_model(str(args.model))
    feature_names = model.get_booster().feature_names
    print(f"  Model expects {len(feature_names)} features", file=sys.stderr)

    # Check which features are available
    available = [f for f in feature_names if f in matched.columns]
    missing = [f for f in feature_names if f not in matched.columns]

    if missing:
        print(f"  Missing features: {missing}", file=sys.stderr)
        print(f"  Available columns: {sorted(matched.columns)}", file=sys.stderr)
        # Try to cast numeric columns
        for col in available:
            if matched[col].dtype == pl.Utf8:
                try:
                    matched = matched.with_columns(pl.col(col).cast(pl.Float64))
                except Exception:
                    pass

    # Ensure all feature columns are numeric
    for col in feature_names:
        if col in matched.columns and matched[col].dtype == pl.Utf8:
            matched = matched.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    # --- Step 5: Score ---
    print("\n=== Step 5: Scoring ===", file=sys.stderr)
    X = matched.select(feature_names)
    print(f"  Feature matrix shape: {X.shape}", file=sys.stderr)

    # Null summary
    null_counts = X.null_count()
    cols_with_nulls = [(col, null_counts[col][0]) for col in feature_names
                       if null_counts[col][0] > 0]
    if cols_with_nulls:
        print(f"  Columns with nulls ({len(cols_with_nulls)}):", file=sys.stderr)
        for col, count in sorted(cols_with_nulls, key=lambda x: -x[1])[:5]:
            pct = count / len(matched) * 100
            print(f"    {col}: {count:,} ({pct:.1f}%)", file=sys.stderr)

    probs = model.predict_proba(X)[:, 1]
    matched = matched.with_columns([
        pl.Series("synthdnm_prob", probs),
        pl.Series("synthdnm_class", (probs >= 0.5).astype(int)),
    ])

    # --- Step 6: Report ---
    print("\n" + "=" * 60, file=sys.stderr)
    print("=== EVALUATION RESULTS ===", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    n_total = len(matched)
    n_real = matched.filter(pl.col("synthdnm_class") == 1).height
    n_artifact = matched.filter(pl.col("synthdnm_class") == 0).height

    print(f"\nTotal matched & scored: {n_total:,}", file=sys.stderr)
    print(f"Classified as real DNM (prob >= 0.5): {n_real:,} ({100*n_real/n_total:.1f}%)",
          file=sys.stderr)
    print(f"Classified as artifact (prob < 0.5): {n_artifact:,} ({100*n_artifact/n_total:.1f}%)",
          file=sys.stderr)

    prob_col = matched["synthdnm_prob"]
    print(f"\nsynthdnm_prob distribution:", file=sys.stderr)
    print(f"  Mean:   {prob_col.mean():.6f}", file=sys.stderr)
    print(f"  Median: {prob_col.median():.6f}", file=sys.stderr)
    print(f"  Std:    {prob_col.std():.6f}", file=sys.stderr)

    # Probability buckets
    print(f"\nProbability buckets:", file=sys.stderr)
    for lo, hi, label in [
        (0.0, 0.01, "< 0.01 (very confident artifact)"),
        (0.01, 0.1, "0.01 - 0.1"),
        (0.1, 0.3, "0.1 - 0.3"),
        (0.3, 0.5, "0.3 - 0.5"),
        (0.5, 0.7, "0.5 - 0.7"),
        (0.7, 0.9, "0.7 - 0.9"),
        (0.9, 1.01, ">= 0.9 (very confident real DNM)"),
    ]:
        count = matched.filter(
            (pl.col("synthdnm_prob") >= lo) & (pl.col("synthdnm_prob") < hi)
        ).height
        pct = 100 * count / n_total if n_total > 0 else 0
        print(f"  {label}: {count:,} ({pct:.1f}%)", file=sys.stderr)

    # By validation status
    print(f"\nBy validation status:", file=sys.stderr)
    for val_status in matched["Validation"].unique().sort().to_list():
        subset = matched.filter(pl.col("Validation") == val_status)
        n = len(subset)
        n_class1 = subset.filter(pl.col("synthdnm_class") == 1).height
        mean_prob = subset["synthdnm_prob"].mean()
        print(f"  {val_status}: {n:,} total, {n_class1:,} real DNM "
              f"({100*n_class1/n:.1f}%), mean_prob={mean_prob:.4f}", file=sys.stderr)

    # By study
    print(f"\nBy study:", file=sys.stderr)
    study_agg = (matched.group_by("StudyName")
                 .agg([
                     pl.len().alias("n"),
                     (pl.col("synthdnm_class") == 1).sum().alias("n_real"),
                     pl.col("synthdnm_prob").mean().alias("mean_prob"),
                 ])
                 .sort("n", descending=True))
    for row in study_agg.iter_rows(named=True):
        pct = 100 * row["n_real"] / row["n"]
        print(f"  {row['StudyName']}: {row['n']:,} total, {row['n_real']:,} real DNM "
              f"({pct:.1f}%), mean_prob={row['mean_prob']:.4f}", file=sys.stderr)

    # By variant type
    if "indel_flag" in matched.columns:
        print(f"\nBy variant type:", file=sys.stderr)
        for vtype, label in [(0, "SNV"), (1, "Indel")]:
            subset = matched.filter(pl.col("indel_flag") == vtype)
            if len(subset) > 0:
                n = len(subset)
                n_class1 = subset.filter(pl.col("synthdnm_class") == 1).height
                mean_prob = subset["synthdnm_prob"].mean()
                print(f"  {label}: {n:,} total, {n_class1:,} real DNM "
                      f"({100*n_class1/n:.1f}%), mean_prob={mean_prob:.4f}", file=sys.stderr)

    # By function class
    print(f"\nBy function class (top 10):", file=sys.stderr)
    func_agg = (matched.group_by("FunctionClass")
                .agg([
                    pl.len().alias("n"),
                    (pl.col("synthdnm_class") == 1).sum().alias("n_real"),
                    pl.col("synthdnm_prob").mean().alias("mean_prob"),
                ])
                .sort("n", descending=True)
                .head(10))
    for row in func_agg.iter_rows(named=True):
        pct = 100 * row["n_real"] / row["n"]
        print(f"  {row['FunctionClass']}: {row['n']:,} total, {row['n_real']:,} real DNM "
              f"({pct:.1f}%), mean_prob={row['mean_prob']:.4f}", file=sys.stderr)

    # False negative analysis: validated DNMs classified as artifacts
    print(f"\n--- False negatives (validated DNMs classified as artifact) ---", file=sys.stderr)
    fn = matched.filter(
        (pl.col("Validation") == "yes") & (pl.col("synthdnm_class") == 0)
    )
    if fn.shape[0] > 0:
        print(f"  Count: {fn.shape[0]:,}", file=sys.stderr)
        print(f"  Mean prob: {fn['synthdnm_prob'].mean():.4f}", file=sys.stderr)
        # Show some examples
        examples = fn.select(["#CHROM", "POS", "REF", "ALT_specific", "SAMPLE",
                               "synthdnm_prob", "Gene", "FunctionClass"]).head(10)
        print(f"  Examples:", file=sys.stderr)
        print(examples, file=sys.stderr)
    else:
        print(f"  None! All validated DNMs classified correctly.", file=sys.stderr)

    # --- Step 7: Save ---
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.output.endswith(".parquet"):
            matched.write_parquet(args.output)
        else:
            matched.write_csv(args.output, separator="\t")
        print(f"\nWrote {len(matched):,} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
