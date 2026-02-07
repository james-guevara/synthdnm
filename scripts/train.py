#!/usr/bin/env python3
"""
train.py — Train XGBoost classifier for SynthDNM.

Consumes Parquet output from preprocess_features.py and produces a trained
model with evaluation metrics and feature importance.

Feature sets are defined in features.toml and loaded via config.py.

Usage:
    # SSC classifier (GATK INFO features)
    python train.py \
        --input training_data.parquet \
        --output_dir output/models/ \
        --feature_set gatk \
        --cv_folds 5

    # SPARK classifier (FORMAT-derived features only)
    python train.py \
        --input training_data.parquet \
        --output_dir output/models/ \
        --feature_set universal

    # With hyperparameter tuning
    python train.py \
        --input training_data.parquet \
        --output_dir output/models/ \
        --feature_set gatk \
        --tune
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from config import load_config, resolve_feature_set, list_feature_sets, get


def validate_features(
    df: pl.DataFrame,
    features: list[str],
    feature_set_name: str,
) -> tuple[list[str], list[str], list[str]]:
    """Validate requested features against Parquet columns.

    Returns (valid_features, missing_columns, all_null_columns).
    """
    parquet_cols = set(df.columns)
    valid = []
    missing = []
    all_null = []

    for f in features:
        if f not in parquet_cols:
            missing.append(f)
        elif df[f].null_count() == len(df):
            all_null.append(f)
        else:
            valid.append(f)

    return valid, missing, all_null


def print_feature_summary(
    feature_set_name: str,
    features: list[str],
    missing: list[str],
    all_null: list[str],
) -> None:
    """Print feature selection summary to stderr."""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"FEATURE SET: {feature_set_name} ({len(features)} features)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    if missing:
        print(f"\nNot in Parquet (skipped): {', '.join(missing)}", file=sys.stderr)
    if all_null:
        print(f"\nAll null (skipped): {', '.join(all_null)}", file=sys.stderr)

    print(f"\nUsing {len(features)} features:", file=sys.stderr)
    for f in features:
        print(f"  - {f}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)


def train_model(
    X_train: pl.DataFrame,
    y_train: np.ndarray,
    X_test: pl.DataFrame,
    y_test: np.ndarray,
    training_config: dict,
    params: dict | None = None,
) -> xgb.XGBClassifier:
    """Train XGBoost with early stopping on test set.

    Accepts Polars DataFrames directly — XGBoost preserves column names
    as feature names on the booster.
    """
    default_params = {
        "max_depth": training_config["max_depth"],
        "learning_rate": training_config["learning_rate"],
        "n_estimators": training_config["n_estimators"],
        "early_stopping_rounds": training_config["early_stopping_rounds"],
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": training_config["seed"],
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pl.DataFrame,
    y_test: np.ndarray,
    n_train: int = 0,
) -> dict:
    """Evaluate model and print metrics to stderr. Returns metrics dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n{'='*60}", file=sys.stderr)
    print("EVALUATION RESULTS", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"ROC AUC:  {auc:.4f}", file=sys.stderr)
    print(f"Accuracy: {acc:.4f}", file=sys.stderr)
    print(f"\nConfusion Matrix:", file=sys.stderr)
    print(f"  TN={cm[0][0]:>6d}  FP={cm[0][1]:>6d}", file=sys.stderr)
    print(f"  FN={cm[1][0]:>6d}  TP={cm[1][1]:>6d}", file=sys.stderr)
    print(f"\nClassification Report:", file=sys.stderr)
    print(classification_report(y_test, y_pred), file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    best_iteration = model.best_iteration if hasattr(model, "best_iteration") else None

    metrics = {
        "auc": float(auc),
        "accuracy": float(acc),
        "precision": float(report["1"]["precision"]),
        "recall": float(report["1"]["recall"]),
        "f1": float(report["1"]["f1-score"]),
        "best_iteration": best_iteration,
        "n_train": n_train,
        "n_test": int(len(y_test)),
        "confusion_matrix": {
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1]),
        },
    }
    return metrics


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    training_config: dict,
    params: dict | None = None,
) -> dict:
    """Run stratified k-fold cross-validation. Returns per-fold results."""
    default_params = {
        "max_depth": training_config["max_depth"],
        "learning_rate": training_config["learning_rate"],
        "n_estimators": 500,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": training_config["seed"],
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                         random_state=training_config["seed"])
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"CROSS-VALIDATION ({cv_folds}-fold)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for i, score in enumerate(scores):
        print(f"  Fold {i+1}: AUC = {score:.4f}", file=sys.stderr)
    print(f"  Mean AUC: {scores.mean():.4f} +/- {scores.std():.4f}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    return {
        "n_folds": cv_folds,
        "per_fold_auc": [float(s) for s in scores],
        "mean_auc": float(scores.mean()),
        "std_auc": float(scores.std()),
    }


def run_hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    training_config: dict,
    n_iter: int = 20,
    cv_folds: int = 5,
) -> dict:
    """Run randomized hyperparameter search. Returns best params."""
    seed = training_config["seed"]
    param_dist = {
        "max_depth": [3, 4, 5, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }

    base_model = xgb.XGBClassifier(
        n_estimators=500,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=seed,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"\n{'='*60}", file=sys.stderr)
    print("HYPERPARAMETER TUNING RESULTS", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Best AUC: {search.best_score_:.4f}", file=sys.stderr)
    print(f"Best params:", file=sys.stderr)
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    return {k: v if not isinstance(v, np.integer) else int(v)
            for k, v in search.best_params_.items()}


def save_feature_importance(
    model: xgb.XGBClassifier,
    output_path: Path,
) -> None:
    """Save feature importance (gain) as sorted TSV.

    Feature names are already set on the booster via fit(feature_names=...).
    """
    importance = model.get_booster().get_score(importance_type="gain")
    rows = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    with open(output_path, "w") as f:
        f.write("feature\tgain\n")
        for name, gain in rows:
            f.write(f"{name}\t{gain:.4f}\n")

    print(f"Saved feature importance to {output_path}", file=sys.stderr)


def main():
    # --- Load config (for feature set names in help text) ---
    config = load_config()
    available_sets = list_feature_sets(config)
    feature_set_help = ", ".join(
        f"{name} ({desc})" for name, desc in available_sets.items()
    )

    parser = argparse.ArgumentParser(
        description="Train XGBoost classifier for SynthDNM"
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Input Parquet file from preprocess_features.py",
    )
    parser.add_argument(
        "--output_dir", required=True, type=Path,
        help="Directory to save model and metrics",
    )
    parser.add_argument(
        "--feature_set", required=True,
        help=f"Feature set to use: {feature_set_help}",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to features.toml (default: auto-detect)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides features.toml)",
    )
    parser.add_argument(
        "--test_size", type=float, default=None,
        help="Fraction of data for test set (overrides features.toml)",
    )
    parser.add_argument(
        "--cv_folds", type=int, default=0,
        help="Number of CV folds (0 = skip CV, default: 0)",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run randomized hyperparameter search before final training",
    )
    parser.add_argument(
        "--tune_iter", type=int, default=20,
        help="Number of random search iterations (default: 20)",
    )
    args = parser.parse_args()

    # --- Reload config with explicit path if given ---
    if args.config:
        config = load_config(args.config)

    # --- Build training config (TOML defaults, CLI overrides) ---
    training_config = dict(config["training"])
    if args.seed is not None:
        training_config["seed"] = args.seed
    if args.test_size is not None:
        training_config["test_size"] = args.test_size

    label_col = get(config, "columns", "label")

    # --- Resolve feature set ---
    features_requested = resolve_feature_set(config, args.feature_set)
    print(f"Feature set '{args.feature_set}': {len(features_requested)} features requested",
          file=sys.stderr)

    # --- Load data ---
    print(f"Loading {args.input}...", file=sys.stderr)
    df = pl.read_parquet(args.input)
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns", file=sys.stderr)
    print(f"  Label distribution: {df[label_col].value_counts().sort(label_col)}",
          file=sys.stderr)

    # --- Validate features against Parquet ---
    features, missing, all_null = validate_features(
        df, features_requested, args.feature_set,
    )
    print_feature_summary(args.feature_set, features, missing, all_null)

    if not features:
        print("ERROR: No valid features. Check input data and feature set.",
              file=sys.stderr)
        sys.exit(1)

    # --- Prepare data ---
    X = df.select(features)
    y = df[label_col].cast(pl.Int32).to_numpy()

    print(f"Feature matrix: {X.shape}", file=sys.stderr)
    null_counts = X.null_count()
    cols_with_nulls = [
        (col, null_counts[col][0])
        for col in features
        if null_counts[col][0] > 0
    ]
    if cols_with_nulls:
        print("Columns with nulls (XGBoost handles natively):", file=sys.stderr)
        for col, count in sorted(cols_with_nulls, key=lambda x: -x[1]):
            pct = count / len(df) * 100
            print(f"  {col}: {count:,} ({pct:.1f}%)", file=sys.stderr)

    # --- Train/test split (need numpy indices for stratified split) ---
    test_size = training_config["test_size"]
    seed = training_config["seed"]
    indices = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y, test_size=test_size, random_state=seed, stratify=y,
    )
    X_train = X[idx_train]
    X_test = X[idx_test]
    print(
        f"Split: {len(y_train):,} train / {len(y_test):,} test "
        f"(pos rate: train={y_train.mean():.3f}, test={y_test.mean():.3f})",
        file=sys.stderr,
    )

    # --- Optional: hyperparameter tuning ---
    # sklearn's RandomizedSearchCV/cross_val_score need numpy arrays
    best_params = None
    if args.tune:
        print("\nRunning hyperparameter tuning...", file=sys.stderr)
        best_params = run_hyperparameter_tuning(
            X_train.to_numpy(), y_train,
            training_config=training_config,
            n_iter=args.tune_iter,
            cv_folds=args.cv_folds if args.cv_folds > 0 else 5,
        )

    # --- Train final model ---
    print("\nTraining final model...", file=sys.stderr)
    model = train_model(
        X_train, y_train, X_test, y_test,
        training_config=training_config,
        params=best_params,
    )

    # --- Evaluate ---
    metrics = evaluate_model(model, X_test, y_test, n_train=len(y_train))
    metrics["feature_set"] = args.feature_set
    metrics["features"] = features
    metrics["n_features"] = len(features)
    metrics["seed"] = seed
    metrics["test_size"] = test_size
    if missing:
        metrics["missing_features"] = missing
    if all_null:
        metrics["all_null_features"] = all_null
    if best_params:
        metrics["tuned_params"] = best_params

    # --- Optional: cross-validation ---
    # sklearn CV needs numpy arrays
    cv_results = None
    if args.cv_folds > 0:
        cv_results = run_cross_validation(
            X.to_numpy(), y, cv_folds=args.cv_folds,
            training_config=training_config,
            params=best_params,
        )

    # --- Save artifacts ---
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.output_dir / "model.json"
    model.save_model(str(model_path))
    print(f"Saved model to {model_path}", file=sys.stderr)

    importance_path = args.output_dir / "feature_importance.tsv"
    save_feature_importance(model, importance_path)

    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}", file=sys.stderr)

    if cv_results:
        cv_path = args.output_dir / "cv_results.json"
        with open(cv_path, "w") as f:
            json.dump(cv_results, f, indent=2)
        print(f"Saved CV results to {cv_path}", file=sys.stderr)

    print(f"\nDone! AUC={metrics['auc']:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
