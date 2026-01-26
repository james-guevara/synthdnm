// Module: Swap pedigree for synthetic DNM generation
// Pairs families and swaps offspring to create synthetic parent-child relationships

process SWAP_PEDIGREE {
    tag "swap_pedigree"
    conda "${params.python_env}"

    input:
    path ped_file

    output:
    path "${ped_file.baseName}.swapped.psam", emit: swapped_ped

    script:
    """
    python ${projectDir}/scripts/swap_pedigree.py ${ped_file} -o ${ped_file.baseName}.swapped.psam
    """
}
