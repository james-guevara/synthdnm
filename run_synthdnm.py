import argparse
# import datetime
# import numpy as np
# import os
import pandas as pd
from pathlib import Path
# from pybedtools import BedTool
# import sys
# from time import gmtime, strftime
from classify import run_snv_classifier
from classify import run_indel_classifier
from extract_features import make_pedigree_dicts
from extract_features import make_sample_index_dicts
from extract_features import make_offspring_index_dict
from extract_features import make_features_dict 
from extract_features import make_func_name_dict 
from make_private_vcf import make_private_vcf
from swap_pedigree import swap_pedigree
from training import train_snv_classifier
import sys

# todo: provide default feature files for standard VCF formats (GATK, DeepVariant...), allow for making feature file
# Motivation behind using rare, inherited variants: chances of 0/1 parent 0/1 child in only one family being false positive is very low. (do naive probability calculation) vs. if it's a common variant (present in many families). So chances are the rare, inherited  variant is genotyped correctly is relatively high


# What should feature extract file look like?
# 

# current_time = strftime("%Y-%m-%d_%H.%M.%S", gmtime())

def run_classify(args):
    pedigree_dict, offspring_parents_dict, sample_sex_and_phenotype_dict = make_pedigree_dicts(args.ped_file)

    sample_index_dict, index_sample_dict = make_sample_index_dicts(args.vcf_file)
    offspring_index_id_dict = make_offspring_index_dict(offspring_parents_dict, sample_index_dict)

    func_name_dict = make_func_name_dict()
    # Extract features 
    df_dnm_features_dict = make_features_dict(args.vcf_file, offspring_index_id_dict, offspring_parents_dict, sample_index_dict, func_name_dict, args.features_file, args.region)
    df_dnm_features_dict.to_csv("df_dnm_features_dict_testing.tsv", sep = "\t", index = False)

    # Run classifiers...


def run_make_training_set(args):
    pedigree_dict, offspring_parents_dict, sample_sex_and_phenotype_dict = make_pedigree_dicts(args.ped_file)
    sample_index_dict, index_sample_dict = make_sample_index_dicts(args.vcf_file)
    offspring_index_id_dict = make_offspring_index_dict(offspring_parents_dict, sample_index_dict)

    # Creating the swapped pedigree file
    swapped_pedigree_dict = swap_pedigree(pedigree_dict)
    swapped_pedigree_output_file = "{}_swapped.ped".format(Path(args.ped_file).stem)
    with open(swapped_pedigree_output_file, "w") as f:
        for family in swapped_pedigree_dict:
            for offspring in swapped_pedigree_dict[family]["offspring"]: print("{}\t{}\t{}\t{}\t{}\t{}".format(family, offspring, swapped_pedigree_dict[family]["father_id"], swapped_pedigree_dict[family]["mother_id"], sample_sex_and_phenotype_dict[offspring]["sex_code"], sample_sex_and_phenotype_dict[offspring]["phenotype_code"]), file = f)
            father_id = swapped_pedigree_dict[family]["father_id"]
            print("{}\t{}\t{}\t{}\t{}\t{}".format(family, father_id, "0", "0", sample_sex_and_phenotype_dict[father_id]["sex_code"], sample_sex_and_phenotype_dict[father_id]["phenotype_code"]), file = f)
            mother_id = swapped_pedigree_dict[family]["mother_id"]
            print("{}\t{}\t{}\t{}\t{}\t{}".format(family, mother_id, "0", "0", sample_sex_and_phenotype_dict[mother_id]["sex_code"], sample_sex_and_phenotype_dict[mother_id]["phenotype_code"]), file = f)

    swapped_pedigree_dict, swapped_offspring_parents_dict, sample_sex_and_phenotype_dict = make_pedigree_dicts(swapped_pedigree_output_file)

    # Make VCF of private (in 1 family), inherited variants
    make_private_vcf(args.vcf_file, pedigree_dict, sample_index_dict, args.region)

    func_name_dict = make_func_name_dict()
    # Extract features using the swapped pedigree file and the private, inherited VCF 
    df_dnm_features_dict_truth1 = make_features_dict("private.vcf.gz", offspring_index_id_dict, swapped_offspring_parents_dict, sample_index_dict, func_name_dict, args.features_file, args.region)
    df_dnm_features_dict_truth1["truth"] = 1 # These swapped variants will be our true positives

    # For false positives, use the original pedigree file
    df_dnm_features_dict_truth0 = make_features_dict(args.vcf_file, offspring_index_id_dict, offspring_parents_dict, sample_index_dict, func_name_dict, args.features_file, args.region)
    df_dnm_features_dict_truth0["truth"] = 0

    df_dnm_features_concat = pd.concat([df_dnm_features_dict_truth1, df_dnm_features_dict_truth0])
    df_dnm_features_concat.to_csv("df_dnm_features_dict_priv.tsv", sep = "\t", index = False)

    # Train machine learning models with default parameters (by default)
    # User can use grid search mode to perform grid search to find optimal training parameters
    # train_snv_classifier("df_dnm_features_dict_priv.tsv")

# There are 2 modes to synthdnm: classify mode and train mode
# Classify mode consists of:
#   - feature extraction
#   - classification
# Train mode consists of:
#   - create swapped pedigree file
#   - create private (inherited) VCF file
#   - make synthetic de novos by extracting features from the private (inherited) VCF file using the swapped pedigrees, which will be the true positive de novos in the training set
#   - using the original putative de novos as FPs in our training set (or randomly sample them) 
#   - train SNV and indel classifiers on these datasets


parser = argparse.ArgumentParser(description = "SynthDNM: a de novo mutation classifier and training paradigm")
subparsers = parser.add_subparsers(help = "Available sub-commands")

# "Classify" mode arguments
parser_classify = subparsers.add_parser("classify", help = "Classify DNMs using pre-trained classifiers.")
# parser_classify.add_argument("--clf_folder", help = "Folder that contains the classifiers, which must be in .pkl format (if not specified, will look for them in the default data folder)")
parser_classify.set_defaults(func = run_classify)

# "make_training_set" mode arguments
parser_train = subparsers.add_parser("make_training_set", help = "Make training set.")
parser_train.set_defaults(func = run_make_training_set)

# Arguments common to both modes:
parser.add_argument("--vcf_file", help = "VCF file input", required = True)
parser.add_argument("--ped_file", help = "Pedigree file (.fam/.ped/.psam) input", required = True)
# parser.add_argument("--swapped_ped_file", help = "Pre-existing swapped pedigree file", required = True)
parser.add_argument("--region", help = "Interval ('{}' or '{}:{}-{}' in format of chr or chr:start-end) on which to run training or classification")
parser.add_argument("--features_file", help = "Features file input")
parser.add_argument("--output_folder", help = "Output folder for output files (if not used, then output folder is set to 'synthdnm_output')")

args = parser.parse_args()
args.func(args)

"""
To do:
    1. If feature file isn't specified, then extract all the INFO features and FORMAT features using the VCF header, as well as the "default" CUSTOM features.
    2. Change name of private.vcf file (to match the input VCF filename but with "private" appended).
"""
