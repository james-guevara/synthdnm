import joblib
import numpy as np
import pandas as pd
import pybedtools
import pysam
import re
import sys
from typing import Tuple

# Make a few dictionaries/tables from the pedigree file
def make_pedigree_dicts(ped_filepath):
    pedigree_dict = {}
    offspring_parents_dict = {}
    sample_sex_and_phenotype_dict = {}

    with open(ped_filepath, "r") as f:
        for line in f:
            linesplit = line.rstrip().split("\t")
            family_id = linesplit[0]
            sample_id = linesplit[1]
            father_id = linesplit[2]  
            mother_id = linesplit[3]  
            sex_code = linesplit[4]       # ('1' = male, '2' = female, '0' = unknown)
            phenotype_code = linesplit[5] # ('1' = control, '2' = case, '-9'/'0'/non-numeric = missing data if case/control)

            if family_id not in pedigree_dict: pedigree_dict[family_id] = {"father_id": "", "mother_id": "", "offspring": []}
            if father_id == "0" and mother_id == "0": 
                if sex_code == "1": pedigree_dict[family_id]["father_id"] = sample_id
                elif sex_code == "2": pedigree_dict[family_id]["mother_id"] = sample_id 
            else: 
                pedigree_dict[family_id]["offspring"].append(sample_id)
                offspring_parents_dict[sample_id] = (father_id, mother_id)
            sample_sex_and_phenotype_dict[sample_id] = {"sex_code": sex_code, "phenotype_code": phenotype_code}

    return pedigree_dict, offspring_parents_dict, sample_sex_and_phenotype_dict

def make_sample_index_dicts(vcf_filepath):
    vcf_iterator = pysam.VariantFile(vcf_filepath, mode = "r")
    index_sample_dict = dict(enumerate(vcf_iterator.header.samples))
    sample_index_dict = {v: k for k, v in index_sample_dict.items()}
    return sample_index_dict, index_sample_dict

def make_offspring_index_dict(offspring_parents_dict, sample_index_dict):
    offspring_index_id_dict = {}
    for sample in offspring_parents_dict:
        offspring_index_id_dict[sample_index_dict[sample]] = sample 
    return offspring_index_id_dict

def get_ratio(feature):
    # Check that it's tuple of size 2
    buffer_ = 1.0
    return float(feature[0]) / (float(feature[1]) + buffer_)

def get_log2_ratio(feature):
    # Check that it's a tuple of size 2
    buffer_ = 1.0
    feature_log2_coverage_ratio = np.log2( (float(feature[0]) + float(feature[1]) + buffer_) / (np.median([float(feature[0]), float(feature[1])]) + buffer_) )
    return feature_log2_coverage_ratio

# Extract putative de novo mutations from the VCF
def make_features_dict(vcf_filepath, offspring_index_id_dict, offspring_parents_dict, sample_index_dict, sample_sex_and_phenotype_dict, features_file = None, region = None):
    func_name_dict = {"get_ratio": get_ratio, "get_log2_ratio": get_log2_ratio}

    dnm_features_dict = {}
    vcf_iterator = pysam.VariantFile(vcf_filepath, mode = "r")

    excluded_info_features = set(["AC", "AF", "AN", "ClippingRankSum", "DP", "DS", "END", "ExcessHet", "HaplotypeScore", "MLEAC", "MLEAF", "NEGATIVE_TRAIN_SITE", "POSITIVE_TRAIN_SITE", "RAW_MQ", "VariantType", "culprit"])

    info_features = []
    format_features = []
    format_features_for_custom_features = [] # custom features may use these features as inputs (but they should not be saved in the output table)
    custom_features = []
    custom_feature_function_dict = {}
    custom_feature_input_dict = {}
    if features_file: 
        with open(features_file, "r") as f:
            # Check that these features exist in the header...
            info_features = f.readline().replace(" ", "").strip().split(",")
            format_features = f.readline().replace(" ", "").strip().split(",")
            for line in f:
                regex_match = re.match("(^\w+):(int|float)=(\w+)\((\w+)\)", line.rstrip().replace(" ", ""))
                feature_name = regex_match.group(1)
                return_type = regex_match.group(2)
                function_name = regex_match.group(3)
                input_feature_name = regex_match.group(4)
                func = func_name_dict[function_name]
                custom_feature_function_dict[feature_name] = func
                custom_feature_input_dict[feature_name] = input_feature_name

                if input_feature_name not in format_features: format_features_for_custom_features.append(input_feature_name)
    else:
        # Default features to extract
        for info_field in vcf_iterator.header.info:
            if info_field in excluded_info_features: continue
            if vcf_iterator.header.info[info_field].number != 1: continue # Exclude non-scalar values for now
            info_features.append(info_field)
        for format_field in vcf_iterator.header.formats:
            if format_field == "GT": continue
            format_features.append(format_field)
        if "AD" in format_features:
            custom_features.append("AR")
            custom_feature_function_dict["AR"] = func_name_dict["get_ratio"]
            custom_feature_input_dict["AR"] = "AD" 
            custom_features.append("AD_log2_coverage_ratio")
            custom_feature_function_dict["AD_log2_coverage_ratio"] = func_name_dict["get_log2_ratio"]
            custom_feature_input_dict["AD_log2_coverage_ratio"] = "AD" 

    number_of_samples = len(vcf_iterator.header.samples)

    # When we get the format-level features, we want to determine what types (of numbers) they are using the VCF header.
    # If a particular format feature is single-valued (1), then we'll get 3 values for it (for offspring, mother, father).
    # If a particular format feature has one value for each possible genotype ("G"), then we'll get 3 values per sample (and so 9 total for a trio pedigree).
    # For each format feature, get the offspring and their parents' values
    format_features_dict = {}
    vcf_header = vcf_iterator.header
    for record in vcf_iterator:
        # Skip multiallelic variants
        if len(record.alts) > 1: continue
        # Skip X and Y for now (not anymore...)
        # if "X" in record.chrom or "Y" in record.chrom: continue
        # Get INFO-level features
        dnm_info_features = {}
        for feature in info_features:
            if feature in record.info:
                dnm_info_features[feature] = record.info[feature]
            else: dnm_info_features[feature] = np.nan # In error log, say that this feature doesn't exist for this record

        for i in range(number_of_samples):
            if i not in offspring_index_id_dict: continue 
            offspring_id = offspring_index_id_dict[i]
            (father_id, mother_id) = offspring_parents_dict[offspring_id]
            offspring_GT = record.samples[i]["GT"]
            father_GT = record.samples[sample_index_dict[father_id]]["GT"]
            mother_GT = record.samples[sample_index_dict[mother_id]]["GT"]
            # Filter out non-putative de novo mutations
            if "X" in record.chrom and sample_sex_and_phenotype_dict[offspring_id] == "1":
                if offspring_GT != (0,1): continue
                if mother_GT != (0, 0): continue
            elif "Y" in record.chrom and sample_sex_and_phenotype_dict[offspring_id] == "1":
                if offspring_GT != (0,1): continue
                if father_GT != (0, 0): continue
            else:
                if offspring_GT != (0, 1): continue
                if father_GT != (0, 0): continue
                if mother_GT != (0, 0): continue

            key = (record.chrom, record.pos, record.ref, record.alts[0], offspring_id, sample_sex_and_phenotype_dict[offspring_id]["sex_code"], sample_sex_and_phenotype_dict[offspring_id]["phenotype_code"])
            dnm_features_dict[key] = {}

            dnm_features_dict[key]["offspring_GT"] = offspring_GT
            dnm_features_dict[key]["father_GT"] = father_GT 
            dnm_features_dict[key]["mother_GT"] = mother_GT 

            # Getting format fields
            for format_feature in format_features:
                dnm_features_dict[key]["{}_{}".format("offspring", format_feature)] = record.samples[i][format_feature] 
                dnm_features_dict[key]["{}_{}".format("father", format_feature)] = record.samples[sample_index_dict[father_id]][format_feature]
                dnm_features_dict[key]["{}_{}".format("mother", format_feature)] = record.samples[sample_index_dict[mother_id]][format_feature]

            for format_feature in format_features_for_custom_features:
                dnm_features_dict[key]["{}_{}".format("offspring", format_feature)] = record.samples[i][format_feature] 
                dnm_features_dict[key]["{}_{}".format("father", format_feature)] = record.samples[sample_index_dict[father_id]][format_feature]
                dnm_features_dict[key]["{}_{}".format("mother", format_feature)] = record.samples[sample_index_dict[mother_id]][format_feature]


            # Add INFO-level features for this row
            # dnm_features_dict[key] |= dnm_info_features
            dnm_features_dict[key].update(dnm_info_features)


    df_dnm_features_dict = pd.DataFrame.from_dict(dnm_features_dict).transpose().reset_index() 
    for feature_name in custom_features:
        func = custom_feature_function_dict[feature_name]
        input_feature_name = custom_feature_input_dict[feature_name]
        df_dnm_features_dict["offspring_{}".format(feature_name)] = df_dnm_features_dict["offspring_{}".format(input_feature_name)].apply(func)
        df_dnm_features_dict["father_{}".format(feature_name)] = df_dnm_features_dict["father_{}".format(input_feature_name)].apply(func)
        df_dnm_features_dict["mother_{}".format(feature_name)] = df_dnm_features_dict["mother_{}".format(input_feature_name)].apply(func)

    # Get min and max of format features
    for feature in format_features:
        if vcf_header.formats[feature].number == 1:
            df_dnm_features_dict["min_parent_{}".format(feature)] = df_dnm_features_dict[["father_{}".format(feature), "mother_{}".format(feature)]].min(axis = 1)
            df_dnm_features_dict["max_parent_{}".format(feature)] = df_dnm_features_dict[["father_{}".format(feature), "mother_{}".format(feature)]].max(axis = 1)
        elif vcf_header.formats[feature].number == "G":
            for i in range(0, 3):
                df_dnm_features_dict["offspring_{}_{}".format(feature, str(i))] = df_dnm_features_dict["offspring_{}".format(feature)].str.get(i)
                df_dnm_features_dict["father_{}_{}".format(feature, str(i))] = df_dnm_features_dict["father_{}".format(feature)].str.get(i)
                df_dnm_features_dict["mother_{}_{}".format(feature, str(i))] = df_dnm_features_dict["mother_{}".format(feature)].str.get(i)
                df_dnm_features_dict["min_parent_{}_{}".format(feature, str(i))] = df_dnm_features_dict[["father_{}_{}".format(feature, str(i)), "mother_{}_{}".format(feature, str(i))]].min(axis = 1)
                df_dnm_features_dict["max_parent_{}_{}".format(feature, str(i))] = df_dnm_features_dict[["father_{}_{}".format(feature, str(i)), "mother_{}_{}".format(feature, str(i))]].max(axis = 1)
        elif vcf_header.formats[feature].number == "R":
            for i in range(0, 2):
                df_dnm_features_dict["offspring_{}_{}".format(feature, str(i))] = df_dnm_features_dict["offspring_{}".format(feature)].str.get(i)
                df_dnm_features_dict["father_{}_{}".format(feature, str(i))] = df_dnm_features_dict["father_{}".format(feature)].str.get(i)
                df_dnm_features_dict["mother_{}_{}".format(feature, str(i))] = df_dnm_features_dict["mother_{}".format(feature)].str.get(i)
                df_dnm_features_dict["min_parent_{}_{}".format(feature, str(i))] = df_dnm_features_dict[["father_{}_{}".format(feature, str(i)), "mother_{}_{}".format(feature, str(i))]].min(axis = 1)
                df_dnm_features_dict["max_parent_{}_{}".format(feature, str(i))] = df_dnm_features_dict[["father_{}_{}".format(feature, str(i)), "mother_{}_{}".format(feature, str(i))]].max(axis = 1)

    for feature in custom_features:
        df_dnm_features_dict["min_parent_{}".format(feature)] = df_dnm_features_dict[["father_{}".format(feature), "mother_{}".format(feature)]].min(axis = 1)
        df_dnm_features_dict["max_parent_{}".format(feature)] = df_dnm_features_dict[["father_{}".format(feature), "mother_{}".format(feature)]].max(axis = 1)

    # Only retaining necessary features now...
    retained_features = []
    for feature in info_features: retained_features.append(feature)
    for feature in format_features:
        if vcf_header.formats[feature].number == 1:
            retained_features.append("{}_{}".format("offspring", feature))
            retained_features.append("{}_{}".format("max_parent", feature))
            retained_features.append("{}_{}".format("min_parent", feature))
        elif vcf_header.formats[feature].number == "G":
            for i in range(0, 3):
                retained_features.append("{}_{}_{}".format("offspring", feature, i))
                retained_features.append("{}_{}_{}".format("max_parent", feature, i))
                retained_features.append("{}_{}_{}".format("min_parent", feature, i))
        elif vcf_header.formats[feature].number == "R":
            for i in range(0, 2):
                retained_features.append("{}_{}_{}".format("offspring", feature, i))
                retained_features.append("{}_{}_{}".format("max_parent", feature, i))
                retained_features.append("{}_{}_{}".format("min_parent", feature, i))
    for feature in custom_features:
            retained_features.append("{}_{}".format("offspring", feature))
            retained_features.append("{}_{}".format("max_parent", feature))
            retained_features.append("{}_{}".format("min_parent", feature))

    df_dnm_features_dict = df_dnm_features_dict.rename(columns = {"level_0": "chrom", "level_1": "pos", "level_2": "ref", "level_3": "alt", "level_4": "iid", "level_5": "sex", "level_6": "phenotype"})
    key = ["chrom", "pos", "ref", "alt", "iid", "sex", "phenotype"]
    df_dnm_features_dict = df_dnm_features_dict[key + retained_features]
    return df_dnm_features_dict

