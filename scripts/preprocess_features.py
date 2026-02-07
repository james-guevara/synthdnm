#!/usr/bin/env python3
"""
Preprocess raw feature TSVs into training-ready dataset.

Reads raw protofeatures from extract_dnm_features.py output,
computes derived training features, and outputs a single
training-ready Parquet file.

Derived features:
- Allele ratio (AR) from AD for child and parents
- min/max parent AR, GQ, DP
- Individual PL components (PL0, PL1, PL2) with multi-allelic handling
- min/max parent PL components

Usage:
    # Sample 50k per class for training
    python preprocess_features.py \\
        --feature_dir output/features/ \\
        --output training_data.parquet \\
        --gatk \\
        --sample_n 50000

    # Process all data (for production scoring)
    python preprocess_features.py \\
        --feature_dir output/features/ \\
        --output all_features.parquet \\
        --gatk
"""

import argparse
from pathlib import Path
import polars as pl
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess raw features into training-ready dataset."
    )
    parser.add_argument("--feature_dir", required=True,
                        help="Directory containing *_real.tsv and *_synthetic.tsv files")
    parser.add_argument("--output", required=True,
                        help="Output file path (.parquet or .tsv)")
    parser.add_argument("--sample_n", type=int, default=None,
                        help="Sample N rows per class for balanced training (default: use all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--gatk", action="store_true",
                        help="Include GATK-specific INFO fields")
    parser.add_argument("--chroms", type=str, default=None,
                        help="Comma-separated list of chromosomes to include (e.g., chr22 or chr1,chr2)")
    parser.add_argument("--psam", type=str, default=None,
                        help="Pedigree file (.psam/.fam) with sex info for haploid_flag. "
                             "Without this, chrY=haploid and everything else=diploid.")
    parser.add_argument("--par", type=str, default=None,
                        help="BED file with pseudoautosomal regions (0-based). "
                             "Variants in PAR regions are set to diploid even on chrX/Y.")
    return parser.parse_args()


def get_schema_overrides():
    """Define explicit dtypes for columns that Polars may infer inconsistently.

    Sparse regions (e.g., chrY) may have all-null values for some INFO columns,
    causing Polars to infer them as String instead of Float64.
    """
    # Columns that should always be read as strings (comma-separated or categorical)
    string_cols = [
        "#CHROM", "REF", "ALT", "ALT_specific", "SAMPLE",
        "child_AD", "father_AD", "mother_AD",
        "child_PL", "father_PL", "mother_PL",
        "child_FT", "father_FT", "mother_FT",
        "child_PGT", "father_PGT", "mother_PGT",
        "child_PID", "father_PID", "mother_PID",
        "INFO_culprit", "INFO_VariantType",
        # These can be tuples for multi-allelic sites (e.g., "(0.034, 0.086)")
        "INFO_AF", "INFO_MLEAC", "INFO_MLEAF",
    ]
    # Everything else numeric — force float for INFO fields that might be all-null
    float_cols = [
        "child_GQ", "father_GQ", "mother_GQ",
        "child_DP", "father_DP", "mother_DP",
        "child_AB", "father_AB", "mother_AB",
        "INFO_DP", "INFO_ExcessHet", "INFO_FS", "INFO_MQ", "INFO_QD",
        "INFO_SOR", "INFO_VQSLOD", "INFO_BaseQRankSum", "INFO_MQRankSum",
        "INFO_ReadPosRankSum", "INFO_AN", "INFO_InbreedingCoeff",
        "INFO_ClippingRankSum", "INFO_HaplotypeScore",
        "INFO_RAW_MQ", "INFO_MQ0",
    ]
    overrides = {}
    for c in string_cols:
        overrides[c] = pl.Utf8
    for c in float_cols:
        overrides[c] = pl.Float64
    return overrides


def read_and_sample(file_list, label, sample_per_file, seed):
    """Read TSV files and optionally sample from each."""
    schema_overrides = get_schema_overrides()
    frames = []
    for i, f in enumerate(file_list):
        df = pl.read_csv(f, separator="\t", null_values=".",
                         infer_schema_length=10000,
                         schema_overrides=schema_overrides)
        if sample_per_file and len(df) > sample_per_file:
            df = df.sample(n=sample_per_file, seed=seed + i)
        frames.append(df)
        if (i + 1) % 50 == 0:
            print(f"  Read {i + 1}/{len(file_list)} files", file=sys.stderr)
    combined = pl.concat(frames)
    combined = combined.with_columns(pl.lit(label).alias("truth"))
    return combined


def compute_derived_features(df):
    """Compute derived training features from raw protofeatures."""

    allele_num = pl.col("allele_num")

    # --- Parse AD into ref/alt depths ---
    # null_on_oob=True handles cases where AD has fewer elements than expected
    for member in ("child", "father", "mother"):
        ad_col = f"{member}_AD"
        ad_split = pl.col(ad_col).str.split(",")
        df = df.with_columns([
            ad_split.list.get(0, null_on_oob=True).cast(pl.Float64)
              .alias(f"{member}_AD_ref"),
            pl.when(allele_num == 2)
              .then(ad_split.list.get(2, null_on_oob=True).cast(pl.Float64))
              .otherwise(ad_split.list.get(1, null_on_oob=True).cast(pl.Float64))
              .alias(f"{member}_AD_alt"),
        ])

    # --- Compute allele ratios ---
    # AR = ref / (alt + 1), clamped to [0, 1] by inverting if > 1
    for member in ("child", "father", "mother"):
        ref = pl.col(f"{member}_AD_ref")
        alt = pl.col(f"{member}_AD_alt")
        raw_ar = ref / (alt + 1.0)
        df = df.with_columns(
            pl.when(raw_ar > 1.0).then(1.0 / raw_ar).otherwise(raw_ar)
              .alias(f"{member}_AR")
        )

    df = df.with_columns([
        pl.min_horizontal("father_AR", "mother_AR").alias("min_AR"),
        pl.max_horizontal("father_AR", "mother_AR").alias("max_AR"),
    ])

    # --- min/max parent GQ, DP ---
    df = df.with_columns([
        pl.min_horizontal("father_GQ", "mother_GQ").alias("min_GQ"),
        pl.max_horizontal("father_GQ", "mother_GQ").alias("max_GQ"),
        pl.min_horizontal("father_DP", "mother_DP").alias("min_DP"),
        pl.max_horizontal("father_DP", "mother_DP").alias("max_DP"),
    ])

    # --- Parse PL into individual components ---
    # Biallelic (allele_num=1): PL0=PL[0], PL1=PL[1], PL2=PL[2]
    # Second ALT (allele_num=2): PL0=PL[0], PL1=PL[3], PL2=PL[5]
    pl_indices = {"PL0": (0, 0), "PL1": (1, 3), "PL2": (2, 5)}

    for member in ("child", "father", "mother"):
        pl_col = f"{member}_PL"
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
        df = df.with_columns([
            pl.min_horizontal(f"father_{comp}", f"mother_{comp}").alias(f"min_{comp}"),
            pl.max_horizontal(f"father_{comp}", f"mother_{comp}").alias(f"max_{comp}"),
        ])

    return df


def correct_haploid_flag(df, psam_path=None, par_path=None):
    """Re-derive haploid_flag based on sex info and PAR regions.

    Without --psam: chrY=haploid, everything else=diploid (conservative default).
    With --psam: males (sex=1) on chrX/Y = haploid.
    With --par: variants in PAR regions are set to diploid regardless.
    """
    chrom = pl.col("#CHROM")
    old_haploid_sum = df["haploid_flag"].sum()

    if psam_path is None:
        # Conservative default: only chrY is haploid
        df = df.with_columns(
            pl.when(chrom.str.contains("Y"))
              .then(pl.lit(1))
              .otherwise(pl.lit(0))
              .alias("haploid_flag")
        )
        new_haploid_sum = df["haploid_flag"].sum()
        print(f"  haploid_flag: chrY=haploid, rest=diploid (no psam). "
              f"{old_haploid_sum} -> {new_haploid_sum} haploid variants",
              file=sys.stderr)
    else:
        # Read pedigree for sex info
        # Detect if file has a header (starts with #)
        with open(psam_path) as fh:
            first_line = fh.readline()
        has_header = first_line.startswith("#")

        if has_header:
            psam = pl.read_csv(psam_path, separator="\t")
        else:
            # Standard .fam format: FID, IID, PAT, MAT, SEX, PHENO
            psam = pl.read_csv(psam_path, separator="\t", has_header=False,
                               new_columns=["FID", "IID", "PAT", "MAT", "SEX", "PHENO"])

        # Find sample ID and sex columns (handles #IID, IID, SAMPLE, etc.)
        id_col = None
        sex_col = None
        for c in psam.columns:
            cl = c.lower().lstrip("#")
            if cl in ("iid", "sample"):
                id_col = c
            elif cl == "sex":
                sex_col = c

        if id_col is None or sex_col is None:
            print(f"  Warning: Could not find ID/SEX columns in {psam_path} "
                  f"(columns: {psam.columns}). Using chrY-only default.",
                  file=sys.stderr)
            df = df.with_columns(
                pl.when(chrom.str.contains("Y"))
                  .then(pl.lit(1))
                  .otherwise(pl.lit(0))
                  .alias("haploid_flag")
            )
        else:
            sex_map = psam.select([
                pl.col(id_col).cast(pl.Utf8).alias("_psam_id"),
                (pl.col(sex_col).cast(pl.Utf8) == "1").alias("_is_male"),
            ])
            n_male = sex_map.filter(pl.col("_is_male")).height

            df = df.join(sex_map, left_on="SAMPLE", right_on="_psam_id", how="left")

            is_sex_chrom = chrom.str.contains("X") | chrom.str.contains("Y")
            df = df.with_columns(
                pl.when(pl.col("_is_male") & is_sex_chrom)
                  .then(pl.lit(1))
                  .otherwise(pl.lit(0))
                  .alias("haploid_flag")
            ).drop("_is_male")

            new_haploid_sum = df["haploid_flag"].sum()
            print(f"  haploid_flag: {n_male} males from psam, "
                  f"haploid on chrX/Y. "
                  f"{old_haploid_sum} -> {new_haploid_sum} haploid variants",
                  file=sys.stderr)

    # Apply PAR correction
    if par_path is not None:
        par = pl.read_csv(par_path, separator="\t", has_header=False)
        par = par.rename({par.columns[0]: "chrom", par.columns[1]: "start",
                          par.columns[2]: "end"})
        par = par.select(["chrom", "start", "end"]).with_columns(
            pl.col("start").cast(pl.Int64), pl.col("end").cast(pl.Int64),
        )

        n_before = df.filter(pl.col("haploid_flag") == 1).height
        pos = pl.col("POS").cast(pl.Int64)

        # Build a single condition for all PAR regions
        # BED is 0-based half-open [start, end), VCF POS is 1-based
        # so: POS > start AND POS <= end
        in_par = pl.lit(False)
        for row in par.iter_rows(named=True):
            in_par = in_par | (
                (chrom == row["chrom"]) &
                (pos > row["start"]) &
                (pos <= row["end"])
            )

        df = df.with_columns(
            pl.when(in_par)
              .then(pl.lit(0))
              .otherwise(pl.col("haploid_flag"))
              .alias("haploid_flag")
        )

        n_after = df.filter(pl.col("haploid_flag") == 1).height
        n_corrected = n_before - n_after
        print(f"  PAR correction: {n_corrected} variants changed to diploid "
              f"({par.height} PAR regions from {par_path})", file=sys.stderr)

    return df


def main():
    args = parse_args()
    feature_dir = Path(args.feature_dir)

    if args.chroms:
        chroms = [c.strip() for c in args.chroms.split(",")]
        real_files = sorted(
            f for c in chroms for f in feature_dir.glob(f"{c}_*_real.tsv")
        )
        synthetic_files = sorted(
            f for c in chroms for f in feature_dir.glob(f"{c}_*_synthetic.tsv")
        )
    else:
        real_files = sorted(feature_dir.glob("*_real.tsv"))
        synthetic_files = sorted(feature_dir.glob("*_synthetic.tsv"))

    if not real_files:
        print(f"No *_real.tsv files found in {feature_dir}", file=sys.stderr)
        sys.exit(1)
    if not synthetic_files:
        print(f"No *_synthetic.tsv files found in {feature_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(real_files)} real and {len(synthetic_files)} synthetic files",
          file=sys.stderr)

    # Determine per-file sample count
    sample_per_file = None
    if args.sample_n:
        # Sample equally from each file (uniform across genomic regions)
        sample_per_file_real = max(1, args.sample_n // len(real_files))
        sample_per_file_synth = max(1, args.sample_n // len(synthetic_files))
        print(f"Sampling ~{sample_per_file_real}/file (real), "
              f"~{sample_per_file_synth}/file (synthetic)", file=sys.stderr)
    else:
        sample_per_file_real = None
        sample_per_file_synth = None

    # Read and optionally sample
    print("Reading real (truth=0) features...", file=sys.stderr)
    real_df = read_and_sample(real_files, label=0,
                              sample_per_file=sample_per_file_real, seed=args.seed)

    print("Reading synthetic (truth=1) features...", file=sys.stderr)
    synth_df = read_and_sample(synthetic_files, label=1,
                               sample_per_file=sample_per_file_synth, seed=args.seed)

    # Final balanced sampling to exact count
    if args.sample_n:
        if len(real_df) > args.sample_n:
            real_df = real_df.sample(n=args.sample_n, seed=args.seed)
        if len(synth_df) > args.sample_n:
            synth_df = synth_df.sample(n=args.sample_n, seed=args.seed)

    print(f"Real: {len(real_df)} rows, Synthetic: {len(synth_df)} rows", file=sys.stderr)

    df = pl.concat([real_df, synth_df])

    # Compute derived features
    print("Computing derived features...", file=sys.stderr)
    df = compute_derived_features(df)

    # Correct haploid_flag
    print("Correcting haploid_flag...", file=sys.stderr)
    df = correct_haploid_flag(df, psam_path=args.psam, par_path=args.par)

    # Select output columns
    id_cols = ["#CHROM", "POS", "REF", "ALT", "ALT_specific", "SAMPLE", "AC", "allele_num"]

    derived_cols = [
        "child_AR", "min_AR", "max_AR",
        "child_GQ", "min_GQ", "max_GQ",
        "child_PL0", "min_PL0", "max_PL0",
        "child_PL1", "min_PL1", "max_PL1",
        "child_PL2", "min_PL2", "max_PL2",
        "child_DP", "min_DP", "max_DP",
        # Keep all three members for AB (already numeric)
        "child_AB", "father_AB", "mother_AB",
    ]

    info_cols = [
        "INFO_DP", "INFO_ExcessHet", "INFO_FS", "INFO_MQ", "INFO_QD",
        "INFO_SOR", "INFO_VQSLOD", "INFO_BaseQRankSum", "INFO_MQRankSum",
        "INFO_ReadPosRankSum",
    ]
    if args.gatk:
        info_cols.extend([
            "INFO_AF", "INFO_AN", "INFO_InbreedingCoeff",
            "INFO_ClippingRankSum", "INFO_HaplotypeScore",
            "INFO_MLEAC", "INFO_MLEAF", "INFO_RAW_MQ", "INFO_MQ0",
            "INFO_DS", "INFO_NEGATIVE_TRAIN_SITE", "INFO_POSITIVE_TRAIN_SITE",
            "INFO_culprit", "INFO_VariantType",
        ])

    flag_cols = ["indel_flag", "haploid_flag"]
    label_col = ["truth"]

    output_cols = id_cols + derived_cols + info_cols + flag_cols + label_col
    df = df.select(output_cols)

    # Write output
    output_path = args.output
    if output_path.endswith(".parquet"):
        df.write_parquet(output_path)
    else:
        df.write_csv(output_path, separator="\t")

    print(f"Wrote {len(df)} rows to {output_path}", file=sys.stderr)

    # Print feature summary
    print("\n--- Feature summary ---", file=sys.stderr)
    print(f"  ID columns: {len(id_cols)}", file=sys.stderr)
    print(f"  Derived features: {len(derived_cols)}", file=sys.stderr)
    print(f"  INFO features: {len(info_cols)}", file=sys.stderr)
    print(f"  Flags: {len(flag_cols)}", file=sys.stderr)
    print(f"  Total columns: {len(output_cols)}", file=sys.stderr)
    print(f"  Null counts:", file=sys.stderr)
    null_counts = df.select(derived_cols + info_cols).null_count()
    for col in null_counts.columns:
        n = null_counts[col][0]
        if n > 0:
            print(f"    {col}: {n} ({100*n/len(df):.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
