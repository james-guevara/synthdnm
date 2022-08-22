import pysam
import sys

def make_private_vcf(vcf_filepath, pedigree_table, sample_index_table):
    vcf_iterator = pysam.VariantFile(vcf_filepath, mode = "r")
    vcf_out = pysam.VariantFile("private.vcf", mode = "w", header = vcf_iterator.header)
    number_of_samples = len(vcf_iterator.header.samples)
    for record in vcf_iterator:
        # Skip multiallelic variants
        if len(record.alts) > 1: continue
        # Skip X and Y for now
        if "X" in record.chrom or "Y" in record.chrom: continue
        GTs = [sample["GT"] for sample in record.samples.values()]
        families_with_alternate_alleles = set()
        for family_id, family_table in pedigree_table.items():
            # The goal is to print out records that are private variants.
            father_GT = record.samples[sample_index_table[family_table["father_id"]]]["GT"]
            mother_GT = record.samples[sample_index_table[family_table["mother_id"]]]["GT"]
            # One and only one of the parents should be (0, 1) (so we use XOR), else we continue. (But will include missing genotypes...)
            if not ((father_GT == (0, 1)) ^ (mother_GT == (0, 1))): continue
            genotypes_of_offspring = [0, 0, 0] # REF, HET, ALT/OTHER
            for offspring in family_table["offspring"]:
                offspring_GT = record.samples[sample_index_table[offspring]]["GT"]
                if (offspring_GT == (0, 0)): genotypes_of_offspring[0] += 1
                elif (offspring_GT == (0, 1)): genotypes_of_offspring[1] += 1
                else: genotypes_of_offspring[2] += 1
            if genotypes_of_offspring[1] == 0: continue
            if genotypes_of_offspring[2] > 0: continue 
            families_with_alternate_alleles.add(family_id)
            if len(families_with_alternate_alleles) > 1: break
        if len(families_with_alternate_alleles) == 1: vcf_out.write(record) 
    vcf_out.close()
    pysam.tabix_index("private.vcf", preset = "vcf", force = True)

"""
sample_to_parental_table, pedigree_table = make_pedigree_table("tutorial.ped")
sample_index_table, index_sample_table = make_sample_to_index_table("tutorial.vcf.gz")
make_private_vcf("tutorial.vcf.gz", pedigree_table, sample_index_table)
"""
