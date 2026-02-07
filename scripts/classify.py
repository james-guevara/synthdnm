#!/usr/bin/env python3
"""
classify.py — Apply trained SynthDNM classifier to candidate DNMs.

Loads a trained XGBoost model and scores candidate variants. Outputs
the original data with appended probability and classification columns.

Label semantics (training: truth=0=putative DNMs, truth=1=synthetic DNMs):
    Synthetic DNMs are inherited variants used as proxies for real DNMs
    (high-quality het calls). Putative DNMs are mostly artifacts.

    synthdnm_prob close to 1 → looks like a real DNM (high-quality, keep)
    synthdnm_prob close to 0 → looks like an artifact (low-quality, discard)

Usage:
    # Score candidates with the GATK model
    python classify.py \\
        --input candidates.parquet \\
        --model output/models/gatk/model.json \\
        --output scored_candidates.tsv

    # Filter to likely real DNMs (prob > 0.5)
    python classify.py \\
        --input candidates.parquet \\
        --model output/models/gatk/model.json \\
        --output real_dnms.tsv \\
        --threshold 0.5

    # Stricter threshold, Parquet output
    python classify.py \\
        --input candidates.parquet \\
        --model output/models/gatk/model.json \\
        --output scored.parquet \\
        --threshold 0.7
"""

import argparse
import json
import sys
from pathlib import Path

import polars as pl
import xgboost as xgb


def main():
    parser = argparse.ArgumentParser(
        description="Apply trained SynthDNM classifier to candidate DNMs"
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Input Parquet file (from preprocess_features.py)",
    )
    parser.add_argument(
        "--model", required=True, type=Path,
        help="Trained model (model.json from train.py)",
    )
    parser.add_argument(
        "--output", required=True, type=str,
        help="Output file path (.parquet or .tsv)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="If set, only output variants with synthdnm_prob >= threshold "
             "(i.e., likely real DNMs). Default: output all with scores.",
    )
    parser.add_argument(
        "--metrics", type=Path, default=None,
        help="Path to metrics.json from train.py. If not provided, "
             "looks for metrics.json next to model.json.",
    )
    args = parser.parse_args()

    # --- Load model ---
    print(f"Loading model from {args.model}...", file=sys.stderr)
    model = xgb.XGBClassifier()
    model.load_model(str(args.model))

    # Get feature names from the model's booster
    feature_names = model.get_booster().feature_names
    print(f"  Model expects {len(feature_names)} features", file=sys.stderr)

    # --- Load metrics for additional context ---
    metrics_path = args.metrics or args.model.parent / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        print(f"  Feature set: {metrics.get('feature_set', 'unknown')}", file=sys.stderr)
        print(f"  Training AUC: {metrics.get('auc', 'N/A'):.4f}", file=sys.stderr)

    # --- Load data ---
    print(f"Loading {args.input}...", file=sys.stderr)
    df = pl.read_parquet(args.input)
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns", file=sys.stderr)

    # --- Validate features ---
    parquet_cols = set(df.columns)
    missing = [f for f in feature_names if f not in parquet_cols]
    if missing:
        print(f"ERROR: Missing features in input: {', '.join(missing)}",
              file=sys.stderr)
        sys.exit(1)

    # Check for nulls in feature columns
    X = df.select(feature_names)
    null_counts = X.null_count()
    cols_with_nulls = [
        (col, null_counts[col][0])
        for col in feature_names
        if null_counts[col][0] > 0
    ]
    if cols_with_nulls:
        print("  Columns with nulls (XGBoost handles natively):", file=sys.stderr)
        for col, count in sorted(cols_with_nulls, key=lambda x: -x[1])[:5]:
            pct = count / len(df) * 100
            print(f"    {col}: {count:,} ({pct:.1f}%)", file=sys.stderr)

    # --- Predict ---
    print("Scoring variants...", file=sys.stderr)
    probs = model.predict_proba(X)[:, 1]

    # Append scores to original data
    df = df.with_columns([
        pl.Series("synthdnm_prob", probs),
        pl.Series("synthdnm_class", (probs >= 0.5).astype(int)),
    ])

    # --- Apply threshold filter ---
    n_total = len(df)
    if args.threshold is not None:
        df = df.filter(pl.col("synthdnm_prob") >= args.threshold)
        n_kept = len(df)
        print(f"  Threshold {args.threshold}: {n_kept:,}/{n_total:,} variants "
              f"pass ({100*n_kept/n_total:.1f}%)", file=sys.stderr)

    # --- Print summary ---
    print(f"\n--- Classification summary ---", file=sys.stderr)
    print(f"  Total scored: {n_total:,}", file=sys.stderr)
    print(f"  Output rows: {len(df):,}", file=sys.stderr)
    prob_col = df["synthdnm_prob"]
    print(f"  synthdnm_prob: median={prob_col.median():.4f}, "
          f"mean={prob_col.mean():.4f}", file=sys.stderr)
    n_class0 = df.filter(pl.col("synthdnm_class") == 0).height
    n_class1 = df.filter(pl.col("synthdnm_class") == 1).height
    print(f"  Predicted artifact (class 0): {n_class0:,}", file=sys.stderr)
    print(f"  Predicted real DNM (class 1): {n_class1:,}", file=sys.stderr)

    # If truth column exists, compare
    if "truth" in df.columns:
        truth = df["truth"]
        pred = df["synthdnm_class"]
        agree = (truth == pred).sum()
        print(f"  Agreement with truth label: {agree:,}/{len(df):,} "
              f"({100*agree/len(df):.1f}%)", file=sys.stderr)

    # --- Write output ---
    if args.output.endswith(".parquet"):
        df.write_parquet(args.output)
    else:
        df.write_csv(args.output, separator="\t")

    print(f"\nWrote {len(df):,} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
