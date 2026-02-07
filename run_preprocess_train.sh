#!/bin/bash
#SBATCH --account=ddp195
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=job_%j_preprocess_train.out
#SBATCH --error=job_%j_preprocess_train.err

set -euo pipefail

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate python3.12_env_default

SCRIPTS=/expanse/projects/sebat1/s3/data/sebat/nf_synthdnm_ssc/scripts
FEATURES=/expanse/projects/sebat1/s3/data/sebat/nf_synthdnm_ssc/output/features
RESOURCES=/expanse/projects/sebat1/s3/data/sebat/nf_synthdnm_ssc/resources
OUTPUT=/expanse/projects/sebat1/s3/data/sebat/nf_synthdnm_ssc/output

# --- Step 1: Preprocess all chromosomes with haploid_flag correction ---
echo "=== PREPROCESSING ALL CHROMOSOMES ==="
python $SCRIPTS/preprocess_features.py \
    --feature_dir $FEATURES \
    --output $OUTPUT/training_data_all.parquet \
    --gatk \
    --sample_n 50000 \
    --psam $RESOURCES/SSC.psam \
    --par $RESOURCES/hg38_par.bed

echo ""
echo "=== TRAINING: universal feature set ==="
python $SCRIPTS/train.py \
    --input $OUTPUT/training_data_all.parquet \
    --output_dir $OUTPUT/models/universal/ \
    --feature_set universal \
    --config $SCRIPTS/features.toml \
    --cv_folds 5

echo ""
echo "=== TRAINING: gatk feature set ==="
python $SCRIPTS/train.py \
    --input $OUTPUT/training_data_all.parquet \
    --output_dir $OUTPUT/models/gatk/ \
    --feature_set gatk \
    --config $SCRIPTS/features.toml \
    --cv_folds 5

echo ""
echo "=== TRAINING: ssc feature set ==="
python $SCRIPTS/train.py \
    --input $OUTPUT/training_data_all.parquet \
    --output_dir $OUTPUT/models/ssc/ \
    --feature_set ssc \
    --config $SCRIPTS/features.toml \
    --cv_folds 5

echo ""
echo "=== ALL DONE ==="
