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
import joblib
from make_private_vcf import make_private_vcf
from swap_pedigree import swap_pedigree
from training import train_random_forest_classifier 
from training import randomized_grid_search 
import sys

# To do: provide default feature files for standard VCF formats (GATK, DeepVariant...), allow for making feature file
# Motivation behind using rare, inherited variants: chances of 0/1 parent 0/1 child in only one family being false positive is very low. (do naive probability calculation) vs. if it's a common variant (present in many families). So chances are the rare, inherited  variant is genotyped correctly is relatively high
# current_time = strftime("%Y-%m-%d_%H.%M.%S", gmtime())

def run_classify(args):
    pedigree_dict, offspring_parents_dict, sample_sex_and_phenotype_dict = make_pedigree_dicts(args.ped_file)

    sample_index_dict, index_sample_dict = make_sample_index_dicts(args.vcf_file)
    offspring_index_id_dict = make_offspring_index_dict(offspring_parents_dict, sample_index_dict)

    func_name_dict = make_func_name_dict()

    # Extract features 
    df_dnm_features_dict = make_features_dict(args.vcf_file, offspring_index_id_dict, offspring_parents_dict, sample_index_dict, sample_sex_and_phenotype_dict, func_name_dict, args.features_file, args.region)
    df_dnm_features_dict.to_csv("{}/df_dnm_features.tsv".format(args.output_folder), sep = "\t", index = False)

    if args.feature_extraction_only:
        print("Created features file. Exiting...")        
        sys.exit(0)

    # Run classifiers...
    # df_snv_preds   = run_snv_classifier(df_snv, clf_snv)  
    # df_indel_preds = run_indel_classifier(df_indel, clf_indel)  
    # df_snv_msc_preds = run_snv_msc_classifier(df_snv_msc, clf_snv_msc)
    # df_indel_msc_preds = run_indel_msc_classifier(df_indel_msc, clf_indel_msc)
    
    # Merge dataframes and output to table and bed file 

def run_make_training_set(args):
    pedigree_dict, offspring_parents_dict, sample_sex_and_phenotype_dict = make_pedigree_dicts(args.ped_file)
    sample_index_dict, index_sample_dict = make_sample_index_dicts(args.vcf_file)
    offspring_index_id_dict = make_offspring_index_dict(offspring_parents_dict, sample_index_dict)

    if args.swapped_ped_file:
        swapped_pedigree_file = args.swapped_ped_file
    else:
        # Creating the swapped pedigree file
        swapped_pedigree_dict = swap_pedigree(pedigree_dict)
        swapped_pedigree_file = "{}/{}_swapped.ped".format(args.output_folder, Path(args.ped_file).stem)
        with open(swapped_pedigree_file, "w") as f:
            for family in swapped_pedigree_dict:
                for offspring in swapped_pedigree_dict[family]["offspring"]: print("{}\t{}\t{}\t{}\t{}\t{}".format(family, offspring, swapped_pedigree_dict[family]["father_id"], swapped_pedigree_dict[family]["mother_id"], sample_sex_and_phenotype_dict[offspring]["sex_code"], sample_sex_and_phenotype_dict[offspring]["phenotype_code"]), file = f)
                father_id = swapped_pedigree_dict[family]["father_id"]
                print("{}\t{}\t{}\t{}\t{}\t{}".format(family, father_id, "0", "0", sample_sex_and_phenotype_dict[father_id]["sex_code"], sample_sex_and_phenotype_dict[father_id]["phenotype_code"]), file = f)
                mother_id = swapped_pedigree_dict[family]["mother_id"]
                print("{}\t{}\t{}\t{}\t{}\t{}".format(family, mother_id, "0", "0", sample_sex_and_phenotype_dict[mother_id]["sex_code"], sample_sex_and_phenotype_dict[mother_id]["phenotype_code"]), file = f)
        if args.swapped_ped_only: 
            print("Created swapped pedigree file. Exiting...")
            sys.exit(0)

    swapped_pedigree_dict, swapped_offspring_parents_dict, sample_sex_and_phenotype_dict = make_pedigree_dicts(swapped_pedigree_file)

    # Make VCF of private (in 1 family), inherited variants
    bgzipped_private_vcf_filepath = make_private_vcf(args.vcf_file, pedigree_dict, sample_index_dict, args.output_folder, args.region)

    func_name_dict = make_func_name_dict()
    # Extract features using the swapped pedigree file and the private, inherited VCF 
    df_dnm_features_dict_truth1 = make_features_dict(bgzipped_private_vcf_filepath, offspring_index_id_dict, swapped_offspring_parents_dict, sample_index_dict, sample_sex_and_phenotype_dict, func_name_dict, args.features_file, args.region)
    df_dnm_features_dict_truth1["truth"] = 1 # These swapped variants will be our true positives

    # For false positives, use the original pedigree file
    df_dnm_features_dict_truth0 = make_features_dict(args.vcf_file, offspring_index_id_dict, offspring_parents_dict, sample_index_dict, sample_sex_and_phenotype_dict, func_name_dict, args.features_file, args.region)
    df_dnm_features_dict_truth0["truth"] = 0

    df_dnm_features_concat = pd.concat([df_dnm_features_dict_truth1, df_dnm_features_dict_truth0])
    df_dnm_features_concat.to_csv("{}/df_dnm_features_training.tsv".format(args.output_folder), sep = "\t", index = False)
    print("Training set {}/df_dnm_features_training.tsv has been created.".format(args.output_folder))

def train(args):
    # The key is always 6 elements: ["chrom", "pos", "ref", "alt", "iid", "sex"]
    key = ["chrom", "pos", "ref", "alt", "iid", "sex", "phenotype"]

    df_train = pd.read_csv(args.training_set_tsv, sep = "\t")
    df_train_male_sex_chromosomes = df_train[(df_train["sex"] == 1) & ( (df_train["chrom"].str.contains("X")) | (df_train["chrom"].str.contains("Y")) )]

    """
    Split into different training sets 
    """

    df_snv_train = df_train[( snv_mask := (df_train["ref"].str.len() == 1) & (df_train["alt"].str.len() == 1) )]
    df_indel_train = df_train[~snv_mask]

    df_snv_male_sex_chromosomes_train = df_train_male_sex_chromosomes[( snv_mask := (df_train_male_sex_chromosomes["ref"].str.len() == 1) & (df_train_male_sex_chromosomes["alt"].str.len() == 1) )]
    df_indel_male_sex_chromosomes_train = df_snv_male_sex_chromosomes_train[~snv_mask]

    try:
        clf_snv = train_random_forest_classifier(df_snv_train) # SNV classifier uses hyperparameters specified in function definition (by default) 
        joblib.dump(clf_snv, "{}/clf_snv.pkl".format(args.output_folder))
    except ValueError: print("Couldn't create autosome SNV model.")

    try:
        clf_indel = train_random_forest_classifier(df_indel_train)
        joblib.dump(clf_indel, "{}/clf_indel.pkl".format(args.output_folder))
    except ValueError: print("Couldn't create autosome indel model.")

    try:
        clf_snv_male_sex_chromosomes = train_random_forest_classifier(df_snv_male_sex_chromosomes_train)
        joblib.dump(clf_snv_male_sex_chromosomes, "{}/clf_snv_msc.pkl".format(args.output_folder))
    except ValueError: print("Couldn't create SNV male sex chromosome model.")
    try:
        clf_indel_male_sex_chromosomes = train_random_forest_classifier(df_indel_male_sex_chromosomes_train)
        joblib.dump(clf_indel_male_sex_chromosomes, "{}/clf_indel_msc.pkl".format(args.output_folder))
    except ValueError: print("Couldn't create indel male sex chromosome model.")

def grid_search(args):
    key = ["chrom", "pos", "ref", "alt", "iid"]

    df_train = pd.read_csv(args.training_set_tsv, sep = "\t")

    df_snv_train = df_train[( mask := (df_train["ref"].str.len() == 1) & (df_train["alt"].str.len() == 1) )]
    grid_search_snv = randomized_grid_search(df_snv_train) # SNV classifier uses hyperparameters specified in function definition (by default) 

    df_indel_train = df_train[~mask]
    grid_search_indel = randomized_grid_search(df_indel_train)

    #print(grid_search_snv)
    # print(grid_search_snv.best_estimator_)
    # print(grid_search_indel.best_estimator_)
    # print(grid_search_snv.best_score_)
    # print(grid_search_snv.best_params_)
    # print(grid_search_snv.cv_results_)
    df_snv_results = pd.concat([ pd.DataFrame(grid_search_snv.cv_results_["params"]),
                                 pd.DataFrame(grid_search_snv.cv_results_["mean_test_score"], columns = ["Accuracy"]) 
                               ], axis = 1)
    df_snv_results.to_csv("{}/df_snv_grid_search_results.tsv".format(args.output_folder), sep = "\t", index = False)

    try:
        joblib.dump(grid_search_snv.best_estimator_, "{}/clf_snv.pkl".format(args.output_folder))
    except ValueError: print("Couldn't create autosome SNV model.")

    df_indel_results = pd.concat([ pd.DataFrame(grid_search_indel.cv_results_["params"]),
                                 pd.DataFrame(grid_search_indel.cv_results_["mean_test_score"], columns = ["Accuracy"]) 
                               ], axis = 1)
    df_indel_results.to_csv("{}/df_indel_grid_search_results.tsv".format(args.output_folder), sep = "\t", index = False)

    try:
        joblib.dump(grid_search_indel.best_estimator_, "{}/clf_indel.pkl".format(args.output_folder))
    except ValueError: print("Couldn't create autosome indel model.")

"""
There are 3 modes to synthdnm: classify mode, make_training_set mode, and train mode
classify mode consists of:
  - feature extraction
  - classification
make_training_set mode consists of:
  - create swapped pedigree file
  - create private (inherited) VCF file
  - make synthetic de novos by extracting features from the private (inherited) VCF file using the swapped pedigrees, which will be the true positive de novos in the training set
  - using the original putative de novos as FPs in our training set (or randomly sample them) 
train mode consists of:
  - train SNV and indel classifiers on datasets (obtained using make_training_set mode) 
"""

parser = argparse.ArgumentParser(description = "SynthDNM: a de novo mutation classifier and training paradigm")
subparsers = parser.add_subparsers(help = "Available sub-commands")

parser_classify = subparsers.add_parser("classify", help = "Classify DNMs using pre-trained classifiers.")
parser_classify.add_argument("--clf_folder", help = "Folder that contains the classifiers, which must be in .pkl format (if not specified, will look for them in the default data folder)")
parser_classify.set_defaults(func = run_classify)
parser_classify.add_argument("-feature_extraction_only", action = "store_true", help = "Only output the features file (without classifying")

parser_make_training_set = subparsers.add_parser("make_training_set", help = "Make training set.")
parser_make_training_set.set_defaults(func = run_make_training_set)
parser_make_training_set_meg = parser_make_training_set.add_mutually_exclusive_group()
parser_make_training_set_meg.add_argument("-swapped_ped_only", action = "store_true", help = "Only output the swapped pedigree file")
parser_make_training_set_meg.add_argument("--swapped_ped_file", help = "Pre-existing swapped pedigree file")

parser_train = subparsers.add_parser("train", help = "Train classifiers")
parser_train.set_defaults(func = train)

parser_grid_search = subparsers.add_parser("grid_search", help = "Randomized grid search across hyperparameters.")
parser_grid_search.set_defaults(func = grid_search)

# Common arguments:
parser.add_argument("--vcf_file", help = "VCF file input", required = "classify" in sys.argv or "make_training_set" in sys.argv)
parser.add_argument("--ped_file", help = "Pedigree file (.fam/.ped/.psam) input", required = True)
parser.add_argument("--region", help = "Interval ('{}' or '{}:{}-{}' in format of chr or chr:start-end) on which to run training or classification")
parser.add_argument("--features_file", help = "Features file input")
parser.add_argument("--output_folder", help = "Output folder for output files (if not used, then output folder is set to 'synthdnm_output')", type = Path, default = Path("synthdnm_output"))
parser.add_argument("--training_set_tsv", help = "Training set file (created using make_training_set mode)", required = "train" in sys.argv or "grid_search" in sys.argv)

args = parser.parse_args()

args.output_folder.mkdir(parents = True, exist_ok = True)
args.func(args)
