#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// SynthDNM Pipeline
// Generate synthetic de novos for training DNM classifiers
//
// Workflow:
// 1. GET_PRIVATE_VARIANTS - Filter for AC=2 biallelic variants (private to one family)
// 2. TRIO_DNM2_SWAPPED - Run DNM caller with swapped pedigree (synthetic DNMs)
// 3. TRIO_DNM2_REAL - Run DNM caller with real pedigree (putative DNMs)

include { GET_PRIVATE_VARIANTS } from './modules/get_private_variants'
include { TRIO_DNM2 as TRIO_DNM2_SWAPPED } from './modules/trio_dnm2'
include { TRIO_DNM2 as TRIO_DNM2_REAL } from './modules/trio_dnm2'

// ============================================================================
// Parameters
// ============================================================================

// Chromosomes to process
params.chroms = "chr22"

// Input VCFs
params.vcf_dir = null  // Must be set via profile or command line
params.vcf_pattern = "{chrom}.vcf.gz"

// Pedigree files
params.ped_file = null           // Real pedigree (required)
params.swapped_ped_file = null   // Swapped pedigree (required)

// Output
params.outdir = "${projectDir}/output"

// Containers
params.bcftools_container = "/expanse/projects/sebat1/s3/data/sebat/g2mh/scripts/scripts_for_rare_pipeline/bcftools:1.22--h3a4d415_1"

// ============================================================================
// Helper function to build input channel from chromosome list
// ============================================================================

def buildInputChannel(chroms_str, vcf_dir, vcf_pattern) {
    Channel.fromList(chroms_str.tokenize(',')).map { chrom ->
        def vcf_path = vcf_pattern.replace('{chrom}', chrom)
        def vcf_file = file("${vcf_dir}/${vcf_path}")
        def tbi_file = file("${vcf_dir}/${vcf_path}.tbi")
        return tuple(chrom, vcf_file, tbi_file)
    }
}

// ============================================================================
// Main workflow
// ============================================================================

workflow {
    // Validate required parameters
    if (!params.vcf_dir) {
        error "ERROR: params.vcf_dir is required. Use -profile or --vcf_dir"
    }
    if (!params.ped_file) {
        error "ERROR: params.ped_file is required. Use -profile or --ped_file"
    }
    if (!params.swapped_ped_file) {
        error "ERROR: params.swapped_ped_file is required. Use -profile or --swapped_ped_file"
    }

    // Build input channel
    input_vcfs = buildInputChannel(params.chroms, params.vcf_dir, params.vcf_pattern)

    // Step 1: Get private variants (AC=2, biallelic)
    GET_PRIVATE_VARIANTS(input_vcfs)

    // Step 2: Run trio-dnm2 with swapped pedigree (synthetic DNMs)
    swapped_ped = file(params.swapped_ped_file)
    TRIO_DNM2_SWAPPED(
        GET_PRIVATE_VARIANTS.out.private_vcf,
        swapped_ped,
        "swapped"
    )

    // Step 3: Run trio-dnm2 with real pedigree (putative DNMs)
    real_ped = file(params.ped_file)
    TRIO_DNM2_REAL(
        GET_PRIVATE_VARIANTS.out.private_vcf,
        real_ped,
        "real"
    )
}

// ============================================================================
// Entry points for individual steps
// ============================================================================

workflow RUN_GET_PRIVATE_VARIANTS {
    input_vcfs = buildInputChannel(params.chroms, params.vcf_dir, params.vcf_pattern)
    GET_PRIVATE_VARIANTS(input_vcfs)
}

workflow RUN_TRIO_DNM2_SWAPPED {
    // Expects private variants from previous step
    chroms = Channel.fromList(params.chroms.tokenize(','))
    private_vcfs = chroms.map { chrom ->
        tuple(
            chrom,
            file("${params.outdir}/private_variants/${chrom}.private.vcf.gz"),
            file("${params.outdir}/private_variants/${chrom}.private.vcf.gz.csi")
        )
    }
    swapped_ped = file(params.swapped_ped_file)
    TRIO_DNM2_SWAPPED(private_vcfs, swapped_ped, "swapped")
}

workflow RUN_TRIO_DNM2_REAL {
    // Expects private variants from previous step
    chroms = Channel.fromList(params.chroms.tokenize(','))
    private_vcfs = chroms.map { chrom ->
        tuple(
            chrom,
            file("${params.outdir}/private_variants/${chrom}.private.vcf.gz"),
            file("${params.outdir}/private_variants/${chrom}.private.vcf.gz.csi")
        )
    }
    real_ped = file(params.ped_file)
    TRIO_DNM2_REAL(private_vcfs, real_ped, "real")
}
