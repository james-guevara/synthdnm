import argparse
# import datetime
# import numpy as np
# import os
import pandas as pd
from pathlib import Path
from pybedtools import BedTool
# import sys
# from time import gmtime, strftime
from extract_features import make_pedigree_dicts
from extract_features import make_sample_index_dicts
from extract_features import make_offspring_index_dict
from extract_features import make_features_dict 
import joblib
from make_private_vcf import make_private_vcf
from swap_pedigree import swap_pedigree
from training import train_random_forest_classifier 
from training import randomized_grid_search 
import sys

# To do: provide default feature files for standard VCF formats (GATK, DeepVariant...), allow for making feature file
# Motivation behind using rare, inherited variants: chances of 0/1 parent 0/1 child in only one family being false positive is very low. (do naive probability calculation) vs. if it's a common variant (present in many families). So chances are the rare, inherited  variant is genotyped correctly is relatively high
# current_time = strftime("%Y-%m-%d_%H.%M.%S", gmtime())

key = ("chrom", "pos", "ref", "alt", "iid", "sex", "phenotype")

def make_snv_indel_dataframes(df):
    # Make the male sex chromosome dataframe first (to use as a mask for the autosomal df)
    df_msc   = df[( msc_mask := (df["sex"] == 1) & ( (df["chrom"].str.contains("X")) | (df["chrom"].str.contains("Y")) ) )]
    df_autosomal = df[~msc_mask]

    df_snv   = df_autosomal[( snv_mask := (df_autosomal["ref"].str.len() == 1) & (df_autosomal["alt"].str.len() == 1) )]
    df_indel = df_autosomal[~snv_mask]
    df_snv_msc   = df_msc[( snv_mask := (df_msc["ref"].str.len() == 1) & (df_msc["alt"].str.len() == 1) )]
    df_indel_msc = df_msc[~snv_mask]
    return [df_snv.dropna(), df_indel.dropna(), df_snv_msc.dropna(), df_indel_msc.dropna()]

def classify(df, clf_folder, variant_type):
    try:
        clf   = joblib.load("{}/clf_{}.pkl".format(args.clf_folder, variant_type))
        preds = clf.predict_proba(df.drop(list(key), axis = 1).values)
        return preds
    except FileNotFoundError: print("clf_{}.pkl not found in {}; skipping these predictions..".format(variant_type, args.clf_folder))

def concat_preds(df, preds):
    return pd.concat([df, pd.DataFrame(preds, index = df.index)], axis = "columns")

def run_classify(args):
    pedigree_dict, offspring_parents_dict, sample_sex_and_phenotype_dict = make_pedigree_dicts(args.ped_file)

    sample_index_dict, index_sample_dict = make_sample_index_dicts(args.vcf_file)
    offspring_index_id_dict = make_offspring_index_dict(offspring_parents_dict, sample_index_dict)

    # Extract features 
    df_dnm_features = make_features_dict(args.vcf_file, offspring_index_id_dict, offspring_parents_dict, sample_index_dict, sample_sex_and_phenotype_dict, args.features_file, args.region)
    df_dnm_features.to_csv("{}/df_dnm_features.tsv".format(args.output_folder), sep = "\t", index = False)

    if args.feature_extraction_only:
        print("Created features file. Exiting...")        
        sys.exit(0)

    df_snv, df_indel, df_snv_msc, df_indel_msc = make_snv_indel_dataframes(df_dnm_features)

    features_list = list(df_snv.drop(list(key), axis = 1).columns)

    snv_preds       = classify(df_snv, args.clf_folder, "snv")
    indel_preds     = classify(df_indel, args.clf_folder, "indel")
    snv_msc_preds   = classify(df_snv_msc, args.clf_folder, "snv_msc")
    indel_msc_preds = classify(df_indel_msc, args.clf_folder, "indel_msc")

    df_snv_with_preds = concat_preds(df_snv, snv_preds)
    df_indel_with_preds = concat_preds(df_indel, indel_preds)
    df_snv_msc_with_preds = concat_preds(df_snv_msc, snv_msc_preds)
    df_indel_msc_with_preds = concat_preds(df_indel_msc, indel_msc_preds)
    
    df_concat = pd.concat([df_snv_with_preds, df_indel_with_preds, df_snv_msc_with_preds, df_indel_msc_with_preds]) # .sort_values(["chrom", "start", "end"]) # seems unnecessary (perhaps for X and Y?)

    # Print out a bed file with predictions
    f_bed = open("{}/{}".format(args.output_folder, "test.bed"), "w")
    print("\t".join(["chrom", "start", "end"] + list(key)[1:] + ["0", "1"]), file = f_bed)
    for dnm in BedTool.from_dataframe(df_concat[list(key) + [0, 1]]): print("{}\t{}\t{}\t{}".format(dnm.chrom, dnm.start, dnm.end, "\t".join(dnm[1:])), file = f_bed)


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

    # Extract features using the swapped pedigree file and the private, inherited VCF 
    df_dnm_features_dict_truth1 = make_features_dict(bgzipped_private_vcf_filepath, offspring_index_id_dict, swapped_offspring_parents_dict, sample_index_dict, sample_sex_and_phenotype_dict, args.features_file, args.region)
    df_dnm_features_dict_truth1["truth"] = 1 # These swapped variants will be our true positives

    # For false positives, use the original pedigree file
    df_dnm_features_dict_truth0 = make_features_dict(args.vcf_file, offspring_index_id_dict, offspring_parents_dict, sample_index_dict, sample_sex_and_phenotype_dict, args.features_file, args.region)
    df_dnm_features_dict_truth0["truth"] = 0

    df_dnm_features_concat = pd.concat([df_dnm_features_dict_truth1, df_dnm_features_dict_truth0])
    df_dnm_features_concat.to_csv("{}/df_dnm_features_training.tsv".format(args.output_folder), sep = "\t", index = False)
    print("Training set {}/df_dnm_features_training.tsv has been created.".format(args.output_folder))

def train(args):
    def run_train(df, variant_type):
        try:
            clf = train_random_forest_classifier(df) # SNV classifier uses hyperparameters specified in function definition (by default) 
            joblib.dump(clf, "{}/clf_{}.pkl".format(args.output_folder, variant_type))
        except ValueError: print("Couldn't create {} model.".format(variant_type))

    key = ["chrom", "pos", "ref", "alt", "iid", "sex", "phenotype"]

    df_train = pd.read_csv(args.training_set_tsv, sep = "\t")
    """
    Split into different training sets 
    """
    df_snv, df_indel, df_snv_msc, df_indel_msc = make_snv_indel_dataframes(df_train)

    run_train(df_snv, "snv")
    run_train(df_indel, "indel")
    run_train(df_snv_msc, "snv_msc")
    run_train(df_indel_msc, "indel_msc")



def grid_search(args):
    def run_grid_search(df, variant_type):
        try:
            grid_search = randomized_grid_search(df)
            df_results = pd.concat([ pd.DataFrame(grid_search.cv_results_["params"]),
                                     pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns = ["Accuracy"]) 
                                   ], axis = 1)
            df_results.to_csv("{}/df_{}_grid_search_results.tsv".format(args.output_folder, variant_type), sep = "\t", index = False)
            joblib.dump(grid_search.best_estimator_, "{}/clf_{}.pkl".format(args.output_folder, variant_type))
        except ValueError: print("Couldn't create {} model.".format(variant_type))

    key = ["chrom", "pos", "ref", "alt", "iid"]

    df_train = pd.read_csv(args.training_set_tsv, sep = "\t")
    df_snv, df_indel, df_snv_msc, df_indel_msc = make_snv_indel_dataframes(df_train)

    run_grid_search(df_snv, "snv")
    run_grid_search(df_indel, "indel")
    run_grid_search(df_snv_msc, "snv_msc")
    run_grid_search(df_indel_msc, "indel_msc")

"""
There are 4 modes to synthdnm: classify mode, make_training_set mode, and train mode, and grid_search mode
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
parser_classify.add_argument("--clf_folder", help = "Folder that contains the classifiers, which must be in .pkl format (if not specified, will look for them in the default data folder)", required = True)
parser_classify.add_argument("-feature_extraction_only", action = "store_true", help = "Only output the features file (without classifying")
parser_classify.set_defaults(func = run_classify)

parser_make_training_set = subparsers.add_parser("make_training_set", help = "Make training set.")
parser_make_training_set_meg = parser_make_training_set.add_mutually_exclusive_group()
parser_make_training_set_meg.add_argument("-swapped_ped_only", action = "store_true", help = "Only output the swapped pedigree file")
parser_make_training_set_meg.add_argument("--swapped_ped_file", help = "Pre-existing swapped pedigree file")
parser_make_training_set.set_defaults(func = run_make_training_set)

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
