import sys
from random import seed
from random import shuffle
import re
"""
# pedigree_filepath = open(sys.argv[1])
pedigree_filepath = open("tutorial.ped")

# The key is the family_name (FID). The values are a sub-dictionary, with keys father_name, mother_name, and offspring (which is a list).
pedigree_table = {}
sample_sex_and_phenotype_table = {}

for line in pedigree_filepath:
    linesplit = re.split(r'\t|,| ', line.rstrip())
    family_name = linesplit[0]
    sample_name = linesplit[1]
    father_name = linesplit[2]
    mother_name = linesplit[3]
    sex_code = linesplit[4] # ('1' = male, '2' = female, '0' = unknown)
    phenotype_code = linesplit[5]  # ('1' = control, '2' = case, '-9'/'0'/non-numeric = missing data if case/control)

    if family_name not in pedigree_table: pedigree_table[family_name] = {"father_name": "", "mother_name": "", "offspring": []}
    # Handle parents
    # (Later on, I'll add code to check whether father and mother have already been assigned and if so, check that names are equivalent.)
    if father_name == "0" and mother_name == "0": 
        if sex_code == "1": pedigree_table[family_name]["father_name"] = sample_name 
        elif sex_code == "2": pedigree_table[family_name]["mother_name"] = sample_name 
    else:
        pedigree_table[family_name]["offspring"].append(sample_name)

    sample_sex_and_phenotype_table[sample_name] = {"sex_code": sex_code, "phenotype_code": phenotype_code}

# Index the pedigree table
# table_indices = {}
# for index, key in enumerate(pedigree_table):
seed(0)
indices = list(range(len(pedigree_table)))
shuffle(indices)
"""
def swap_pedigree(pedigree_dict):
    seed(0)
    pedigree_keys = list(pedigree_dict)
    pedigree_indices = list(range(len(pedigree_dict)))
    shuffle(pedigree_indices)
    swapped_pedigree_dict = {}
    for i in range(0, len(pedigree_dict), 2):
        family_a = pedigree_keys[pedigree_indices[i]]
        family_a_offspring = pedigree_dict[family_a]["offspring"]
        family_a_father = pedigree_dict[family_a]["father_id"]
        family_a_mother = pedigree_dict[family_a]["mother_id"]

        family_b = pedigree_keys[pedigree_indices[i + 1]]
        family_b_offspring = pedigree_dict[family_b]["offspring"]
        family_b_father = pedigree_dict[family_b]["father_id"]
        family_b_mother = pedigree_dict[family_b]["mother_id"]

        family_a_new_pedigree = {"father_id": family_a_father, "mother_id": family_a_mother, "offspring": family_b_offspring} 
        family_b_new_pedigree = {"father_id": family_b_father, "mother_id": family_b_mother, "offspring": family_a_offspring} 

        swapped_pedigree_dict[family_a] = family_a_new_pedigree     
        swapped_pedigree_dict[family_b] = family_b_new_pedigree     
    return swapped_pedigree_dict


"""
swapped_pedigree_table = {}
pedigree_table_key_list = list(pedigree_table)

for i in range(0, len(pedigree_table), 2):
    # To do: if there are odd number of families, ignore the last one...
    family_a = pedigree_table_key_list[indices[i]]
    family_a_offspring = pedigree_table[family_a]["offspring"]
    family_a_father = pedigree_table[family_a]["father_name"]
    family_a_mother = pedigree_table[family_a]["mother_name"]

    family_b = pedigree_table_key_list[indices[i + 1]]
    family_b_offspring = pedigree_table[family_b]["offspring"]
    family_b_father = pedigree_table[family_b]["father_name"]
    family_b_mother = pedigree_table[family_b]["mother_name"]

    family_a_new_pedigree = {"father_name": family_a_father, "mother_name": family_a_mother, "offspring": family_b_offspring} 
    family_b_new_pedigree = {"father_name": family_b_father, "mother_name": family_b_mother, "offspring": family_a_offspring} 

    swapped_pedigree_table[family_a] = family_a_new_pedigree     
    swapped_pedigree_table[family_b] = family_b_new_pedigree     

# Print out to file (with name e.g. "tutorial_swapped.ped")
swapped_ped = []
for family in swapped_pedigree_table:
    for offspring in swapped_pedigree_table[family]["offspring"]:
        print("{}\t{}\t{}\t{}\t{}\t{}".format(family, offspring, swapped_pedigree_table[family]["father_name"], swapped_pedigree_table[family]["mother_name"], sample_sex_and_phenotype_table[offspring]["sex_code"], sample_sex_and_phenotype_table[offspring]["phenotype_code"]))
    father_name = swapped_pedigree_table[family]["father_name"]
    print("{}\t{}\t{}\t{}\t{}\t{}".format(family, father_name, "0", "0", sample_sex_and_phenotype_table[father_name]["sex_code"], sample_sex_and_phenotype_table[father_name]["phenotype_code"]))
    mother_name = swapped_pedigree_table[family]["mother_name"]
    print("{}\t{}\t{}\t{}\t{}\t{}".format(family, mother_name, "0", "0", sample_sex_and_phenotype_table[mother_name]["sex_code"], sample_sex_and_phenotype_table[mother_name]["phenotype_code"]))
"""
