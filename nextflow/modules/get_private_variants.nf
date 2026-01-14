// Module: Get private variants (AC=2, biallelic)
// These are definitively inherited variants (private to one family)

process GET_PRIVATE_VARIANTS {
    tag "${chrom}"
    cpus 8
    container "${params.bcftools_container}"

    input:
    tuple val(chrom), path(vcf), path(tbi)

    output:
    tuple val(chrom), path("${chrom}.private.vcf.gz"), path("${chrom}.private.vcf.gz.csi"), emit: private_vcf

    script:
    """
    # Filter for:
    #   --min-ac 2 --max-ac 2: exactly 2 alleles in population (private to one family)
    #   --max-alleles 2: biallelic only (ref + 1 alt)
    bcftools view \
        --threads ${task.cpus} \
        --min-ac 2 \
        --max-ac 2 \
        --max-alleles 2 \
        -O z \
        -o ${chrom}.private.vcf.gz \
        ${vcf}

    bcftools index --threads ${task.cpus} ${chrom}.private.vcf.gz
    """
}
