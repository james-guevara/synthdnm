#!/bin/bash
#SBATCH --account=ddp195
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=job_%j_evaluate_denovo_db.out
#SBATCH --error=job_%j_evaluate_denovo_db.err

set -euo pipefail

eval "$(micromamba shell hook --shell bash)"
micromamba activate python3.12_env_default

BASE=/expanse/projects/sebat1/s3/data/sebat/nf_synthdnm_ssc
SSC_JG=/expanse/projects/sebat1/s3/data/sebat/SSC_JG

echo "=== Evaluating SynthDNM classifier against denovo-db ==="

# Run evaluation for each model (universal, gatk, ssc)
for FEATURE_SET in gatk ssc universal; do
    echo ""
    echo "=============================================="
    echo "=== Feature set: $FEATURE_SET ==="
    echo "=============================================="
    python $BASE/scripts/evaluate_denovo_db.py \
        --denovo_db $BASE/resources/denovo-db.ssc-samples.variants.v.1.6.1.hg38.tsv.gz \
        --feature_dir $BASE/output/features/ \
        --phase_ids_dir $SSC_JG/documentation/ \
        --model $BASE/output/models/$FEATURE_SET/model.json \
        --config $BASE/scripts/features.toml \
        --feature_set $FEATURE_SET \
        --output $BASE/output/evaluation/${FEATURE_SET}_denovo_db_eval.tsv
done

echo ""
echo "=== ALL DONE ==="
