#!/bin/bash
#SBATCH --account=ddp195
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --output=logs/extract_real_%a_%j.out
#SBATCH --error=logs/extract_real_%a_%j.err
#SBATCH --job-name=ext_real
#SBATCH --array=1-166

eval "$(micromamba shell hook --shell bash)"
micromamba activate python3.12_env_default

cd /expanse/projects/sebat1/s3/data/sebat/nf_synthdnm_ssc

REGION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" regions_20mb.txt)
# e.g. chr1:1-20000000 -> chr1_1-20000000
REGION_SAFE=$(echo "$REGION" | tr ':' '_')

VCF_CHROM=$(echo "$REGION" | cut -d: -f1)
VCF="/expanse/projects/sebat1/s3/data/sebat/SSC_JG/gatk/${VCF_CHROM}.masked.vcf.gz"
PED="resources/SSC.psam"
OUTDIR="output/features"
mkdir -p $OUTDIR

date
echo "Task ${SLURM_ARRAY_TASK_ID}: Real DNMs for ${REGION}"

python scripts/extract_dnm_features.py \
    --vcf $VCF \
    --ped $PED \
    --real_dnms \
    --gatk \
    --region $REGION \
    --output $OUTDIR/${REGION_SAFE}_real.tsv

date
echo "Done"
