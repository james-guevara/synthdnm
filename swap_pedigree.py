import sys
import re

# pedigree_filepath = open(sys.argv[1])
pedigree_filepath = open("pedigree_test.ped")

for line in pedigree_filepath:
    linesplit = re.split(r'\t|,| ', line.rstrip())
    family_name = linesplit[0]
    sample_name = linesplit[1]
    father_name = linesplit[2]
    mother_name = linesplit[3]
    sex_code = linesplit[4] # ('1' = male, '2' = female, '0' = unknown)
    phenotype_code = linesplit[5]  # ('1' = control, '2' = case, '-9'/'0'/non-numeric = missing data if case/control)
