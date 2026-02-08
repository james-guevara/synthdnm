#!/bin/bash
#SBATCH --account=ddp195
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --output=logs/extract_sex_synth_%a_%j.out
#SBATCH --error=logs/extract_sex_synth_%a_%j.err
#SBATCH --job-name=ext_sex_synth
#SBATCH --array=156-166

eval "$(micromamba shell hook --shell bash)"
micromamba activate python3.12_env_default

cd /expanse/projects/sebat1/s3/data/sebat/nf_synthdnm_ssc

REGION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" regions_20mb.txt)
REGION_SAFE=$(echo "$REGION" | tr ':' '_')

VCF_CHROM=$(echo "$REGION" | cut -d: -f1)
VCF="/expanse/projects/sebat1/s3/data/sebat/SSC_JG/gatk/${VCF_CHROM}.masked.vcf.gz"
PED="resources/SSC.psam"
SWAPPED_PED="resources/SSC.swapped.psam"
OUTDIR="output/features_sex_chrom"
mkdir -p $OUTDIR

date
echo "Task ${SLURM_ARRAY_TASK_ID}: Sex chrom synthetic DNMs for ${REGION}"

python scripts/extract_sex_chrom_dnm_features.py \
    --vcf $VCF \
    --ped $PED \
    --swapped_ped $SWAPPED_PED \
    --gatk \
    --region $REGION \
    --output $OUTDIR/${REGION_SAFE}_synthetic.tsv

date
echo "Done"
