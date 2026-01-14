# SynthDNM Nextflow Pipeline

Nextflow workflow for generating synthetic de novo mutations (DNMs) for training DNM classifiers.

## Overview

This pipeline processes VCF files to create:
1. **Synthetic DNMs** - High-quality inherited variants that appear as DNMs when parents are swapped (positive training set)
2. **Putative DNMs** - Candidate DNMs from real pedigree (to be classified)

## Workflow Steps

1. **GET_PRIVATE_VARIANTS** - Filter for AC=2, biallelic variants (private to one family = definitively inherited)
2. **TRIO_DNM2_SWAPPED** - Run `bcftools +trio-dnm2` with swapped pedigree → synthetic DNMs
3. **TRIO_DNM2_REAL** - Run `bcftools +trio-dnm2` with real pedigree → putative DNMs

## Requirements

- Nextflow >= 24.0.0
- Singularity
- SLURM (configured for SDSC Expanse)

## Usage

```bash
cd nextflow

# Test on chr22
nextflow run main.nf -profile ssc_wgs --chroms chr22

# Run all autosomes
nextflow run main.nf -profile ssc_wgs \
  --chroms 'chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22'

# Run individual steps
nextflow run main.nf -entry RUN_GET_PRIVATE_VARIANTS -profile ssc_wgs
nextflow run main.nf -entry RUN_TRIO_DNM2_SWAPPED -profile ssc_wgs
nextflow run main.nf -entry RUN_TRIO_DNM2_REAL -profile ssc_wgs
```

## Profiles

| Profile | Description |
|---------|-------------|
| `ssc_wgs` | SSC WGS cohort |
| `spark_iwes` | SPARK iWES v3 |
| `spark_iwgs` | SPARK iWGS v1.1 |

## Output

```
output/
├── private_variants/    # chr*.private.vcf.gz
├── synthetic_dnms/      # chr*.swapped_dnms.vcf.gz
└── putative_dnms/       # chr*.real_dnms.vcf.gz
```

## Resources

Per-chromosome processing with dynamic resource allocation:
- Large chromosomes (chr1-3): 48-64 GB memory, 6-8h
- Medium chromosomes (chr4-12): 48 GB memory, 4-6h
- Small chromosomes (chr13-22, X, Y): 32-48 GB memory, 4h

## Next Steps

After running this pipeline, use the Python scripts in the parent directory to:
1. Extract features (`extract_features.py`)
2. Train classifier (`training.py`)
3. Classify putative DNMs (`classify.py`)
