# SynthDNM

Classifier for filtering de novo mutations (DNMs) from whole-genome sequencing trios.

Trains on synthetic DNMs (inherited variants via pedigree swapping) as proxies for real de novos, and putative DNMs (child het, parents hom ref) which are mostly artifacts. The classifier learns to distinguish high-quality real DNMs from noise.

## Pipeline

```
1. swap_pedigree.py        — Generate swapped pedigree
2. generate_regions.py     — Shard genome into regions for parallel extraction
3. extract_dnm_features.py — Extract genotype features (FORMAT + INFO fields)
4. preprocess_features.py  — Derive features, correct haploid_flag, output Parquet
5. train.py                — Train XGBoost classifier (3 feature sets)
6. classify.py             — Score candidate DNMs
```

## Feature Sets

Defined in `scripts/features.toml` with an extends chain:

| Feature Set | Features | Use Case |
|---|---|---|
| **universal** | 21 (FORMAT-only) | Any caller (DeepVariant, DRAGEN, etc.) |
| **gatk** | 28 (+INFO quality metrics) | GATK HaplotypeCaller |
| **ssc** | 29 (+VQSLOD) | SSC-specific GATK with VQSR |

## Label Semantics

- `synthdnm_prob` = P(truth=1) = probability variant is a real DNM
- **High prob** (close to 1) = looks like a real de novo = **keep**
- **Low prob** (close to 0) = looks like artifact = **discard**

Training labels:
- truth=1: Synthetic DNMs — inherited variants used as proxies for real DNMs (high-quality)
- truth=0: Putative DNMs — child het, parents hom ref (vast majority are artifacts)

## Validation

Evaluated against [denovo-db v1.6.1](https://denovo-db.gs.washington.edu/) SSC validated DNMs.
See `results/validation_summary.json` for full metrics.

| Model | Validated DNMs (n=3,006) | + AC<=1 filter (n=68,687) |
|---|---|---|
| universal | 96.8% recalled | 98.7% real |
| gatk | 95.5% recalled | 97.9% real |
| ssc | 96.5% recalled | 98.6% real |

Note: Additional downstream filters (gnomAD AF, segmental duplications, simple repeats) not yet applied.

## Usage

### Extract features (SLURM array jobs)
```bash
sbatch run_extract_real.sh
sbatch run_extract_synthetic.sh
```

### Preprocess + train
```bash
python scripts/preprocess_features.py \
    --feature_dir output/features/ \
    --output output/training_data.parquet \
    --gatk --sample_n 50000 \
    --psam resources/SSC.psam \
    --par resources/hg38_par.bed

python scripts/train.py \
    --input output/training_data.parquet \
    --output_dir output/models/gatk/ \
    --feature_set gatk \
    --config scripts/features.toml \
    --cv_folds 5
```

### Classify candidates
```bash
python scripts/classify.py \
    --input candidates.parquet \
    --model output/models/gatk/model.json \
    --output scored.tsv \
    --threshold 0.5
```

## Requirements

- Python 3.12+
- polars, xgboost, scikit-learn, numpy, cyvcf2
