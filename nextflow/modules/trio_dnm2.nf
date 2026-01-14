// Module: Run bcftools +trio-dnm2 to call de novo mutations
// Can be used with real or swapped pedigree files

process TRIO_DNM2 {
    tag "${chrom}_${ped_type}"
    cpus 8
    container "${params.bcftools_container}"

    input:
    tuple val(chrom), path(vcf), path(csi)
    path ped_file
    val ped_type  // "swapped" or "real" - used for output naming

    output:
    tuple val(chrom), val(ped_type), path("${chrom}.${ped_type}_dnms.vcf.gz"), path("${chrom}.${ped_type}_dnms.vcf.gz.csi"), emit: dnm_vcf

    script:
    """
    # Run trio-dnm2 with NAIVE method (simpler, good for training data)
    # Then filter to keep only variants with DNM annotation
    bcftools +trio-dnm2 \
        --threads ${task.cpus} \
        --use-NAIVE \
        --ped ${ped_file} \
        -O u \
        ${vcf} \
    | bcftools filter \
        --threads ${task.cpus} \
        -i 'DNM!="."' \
        -O z \
        -o ${chrom}.${ped_type}_dnms.vcf.gz

    bcftools index --threads ${task.cpus} ${chrom}.${ped_type}_dnms.vcf.gz
    """
}
