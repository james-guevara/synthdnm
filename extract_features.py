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


def dispatch(func_name_dict, name, *args, **kwargs):
    print(*args)
    func_name_dict[name](*args, **kwargs)

def get_ratio(feature):
    # Check that it's tuple of size 2
    buffer_ = 1.0
    return float(feature[0]) / (float(feature[1]) + buffer_)

def get_log2_ratio(feature):
    # Check that it's a tuple of size 2
    buffer_ = 1.0
    feature_log2_coverage_ratio = np.log2( (float(feature[0]) + float(feature[1]) + buffer_) / (np.median([float(feature[0]), float(feature[1])]) + buffer_) )
    return feature_log2_coverage_ratio

def make_func_name_dict():
    func_name_dict = {"get_ratio": get_ratio, "get_log2_ratio": get_log2_ratio}
    return func_name_dict

# Extract putative de novo mutations from the VCF
def make_features_dict(vcf_filepath, offspring_index_id_dict, offspring_parents_dict, sample_index_dict, func_name_dict, features_file = None):
    dnm_features_dict = {}

    header_id = "\t".join(["chrom", "pos", "ref", "alt", "iid", "offspring_GT", "father_GT", "mother_GT"])

    info_features = []
    format_features = []
    custom_feature_lines = []
    if features_file: 
        with open(features_file, "r") as f:
            # Check that these features exist in the header...
            info_features = f.readline().replace(" ", "").strip().split(",")
            format_features = f.readline().replace(" ", "").strip().split(",")
            for line in f:
                custom_feature_lines.append(line.rstrip())
    else:
        # Default features to extract
        info_features = ["VQSLOD", "ClippingRankSum", "BaseQRankSum", "FS", "SOR", "MQ", "MQRankSum", "QD", "ReadPosRankSum"]
        format_features = ["AD", "DP", "GQ", "PL"]


    vcf_iterator = pysam.VariantFile(vcf_filepath, mode = "r")
    number_of_samples = len(vcf_iterator.header.samples)

    # When we get the format-level features, we want to determine what types (of numbers) they are using the VCF header.
    # If a particular format feature is single-valued (1), then we'll get 3 values for it (for offspring, mother, father).
    # If a particular format feature has one value for each possible genotype ("G"), then we'll get 3 values per sample (and so 9 total for a trio pedigree).
    format_features_dict = {}
    vcf_header = vcf_iterator.header

    #for format_feature in format_features:
    #    number_ = vcf_header.formats[format_feature].number
    #    if number_ == 1: format_features_dict[format_feature] = {"offspring_value": float, "father_value": float, "mother_value": float}
    #    elif number_ == "G": format_features_dict[format_feature] = {"offspring_value": Tuple[float, float, float], "father_value": Tuple[float, float, float], "mother_value": Tuple[float, float, float]}
    #    elif number_ == "R": format_features_dict[format_feature] = {"offspring_value": Tuple[float, float], "father_value": Tuple[float, float], "mother_value": Tuple[float, float]}

    # For each format feature, get the offspring and their parents' values
    for record in vcf_iterator:
        # Skip multiallelic variants
        if len(record.alts) > 1: continue
        # Skip X and Y for now
        if "X" in record.chrom or "Y" in record.chrom: continue
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
            if offspring_GT != (0, 1): continue
            if father_GT != (0, 0): continue
            if mother_GT != (0, 0): continue


            key = (record.chrom, record.pos, record.ref, record.alts[0], offspring_id)
            dnm_features_dict[key] = {}

            dnm_features_dict[key]["offspring_GT"] = offspring_GT
            dnm_features_dict[key]["father_GT"] = father_GT 
            dnm_features_dict[key]["mother_GT"] = mother_GT 

            # Getting custom format fields
            for format_feature in format_features:
                #format_features_dict[format_feature]["offspring_value"] = record.samples[i][format_feature]
                #format_features_dict[format_feature]["father_value"] = record.samples[sample_index_dict[father_id]][format_feature]
                #format_features_dict[format_feature]["mother_value"] = record.samples[sample_index_dict[mother_id]][format_feature]
                dnm_features_dict[key]["{}_{}".format("offspring", format_feature)] = record.samples[i][format_feature] 
                dnm_features_dict[key]["{}_{}".format("father", format_feature)] = record.samples[sample_index_dict[father_id]][format_feature]
                dnm_features_dict[key]["{}_{}".format("mother", format_feature)] = record.samples[sample_index_dict[mother_id]][format_feature]

            # FORMAT-level features
            #for feature, feature_dict in format_features_dict.items():
            #    if vcf_header.formats[feature].number == 1:
            #        dnm_features_dict[key]["{}_{}".format("offspring", feature)] = feature_dict["offspring_value"]
            #        dnm_features_dict[key]["{}_{}".format("father", feature)] = feature_dict["father_value"]
            #        dnm_features_dict[key]["{}_{}".format("mother", feature)] = feature_dict["mother_value"]
            #    elif vcf_header.formats[feature].number == "G": # There's a single value for each genotype (so 3 total)
            #        dnm_features_dict[key]["{}_{}_0".format("offspring", feature)] = feature_dict["offspring_value"][0]
            #        dnm_features_dict[key]["{}_{}_1".format("offspring", feature)] = feature_dict["offspring_value"][1]
            #        dnm_features_dict[key]["{}_{}_2".format("offspring", feature)] = feature_dict["offspring_value"][2]
            #        dnm_features_dict[key]["{}_{}_0".format("father", feature)] = feature_dict["father_value"][0]
            #        dnm_features_dict[key]["{}_{}_1".format("father", feature)] = feature_dict["father_value"][1]
            #        dnm_features_dict[key]["{}_{}_2".format("father", feature)] = feature_dict["father_value"][2]
            #        dnm_features_dict[key]["{}_{}_0".format("mother", feature)] = feature_dict["mother_value"][0]
            #        dnm_features_dict[key]["{}_{}_1".format("mother", feature)] = feature_dict["mother_value"][1]
            #        dnm_features_dict[key]["{}_{}_2".format("mother", feature)] = feature_dict["mother_value"][2]
            #    elif vcf_header.formats[feature].number == "R": # There's a single value for each allele type (so 2 total)
            #        dnm_features_dict[key]["{}_{}_0".format("offspring", feature)] = feature_dict["offspring_value"][0]
            #        dnm_features_dict[key]["{}_{}_1".format("offspring", feature)] = feature_dict["offspring_value"][1]
            #        dnm_features_dict[key]["{}_{}_0".format("father", feature)] = feature_dict["father_value"][0]
            #        dnm_features_dict[key]["{}_{}_1".format("father", feature)] = feature_dict["father_value"][1]
            #        dnm_features_dict[key]["{}_{}_0".format("mother", feature)] = feature_dict["mother_value"][0]
            #        dnm_features_dict[key]["{}_{}_1".format("mother", feature)] = feature_dict["mother_value"][1]

            # Add INFO-level features
            # dnm_features_dict[key] |= dnm_info_features
            dnm_features_dict[key].update(dnm_info_features)

    format_and_custom_features = format_features
    df_dnm_features_dict = pd.DataFrame.from_dict(dnm_features_dict).transpose().reset_index() 
    for custom_feature_line in custom_feature_lines:
        regex_match = re.match("(^\w+):(int|float)=(\w+)\((\w+)\)", custom_feature_line.replace(" ", ""))
        feature_name = regex_match.group(1)
        format_and_custom_features.append(feature_name)
        return_type = regex_match.group(2)
        function_name = regex_match.group(3)
        input_feature_name = regex_match.group(4)
        func = func_name_dict[function_name]
        df_dnm_features_dict["offspring_{}".format(feature_name)] = df_dnm_features_dict["offspring_{}".format(input_feature_name)].apply(func)
        df_dnm_features_dict["father_{}".format(feature_name)] = df_dnm_features_dict["father_{}".format(input_feature_name)].apply(func)
        df_dnm_features_dict["mother_{}".format(feature_name)] = df_dnm_features_dict["mother_{}".format(input_feature_name)].apply(func)

    # Now, simply get the max and min of the mother and father features...

    return df_dnm_features_dict


def postprocess_df(df_filepath, features_file, vcf_filepath):
    function_dispatcher = {
            "get_ratio": get_ratio,
            "get_log2_ratio": get_log2_ratio
            }

    vcf_header = pysam.VariantFile(vcf_filepath, mode = "r").header
    df = pd.read_csv(df_filepath, sep = "\t")
    # Read the features file to get the format-level features. Keep the offspring values, apply some functions (min, max) to the parents' values.
    info_features = None
    format_features = None
    custom_features = None
    with open(features_file, "r") as f:
        info_features = f.readline().replace(" ", "").strip().split(",")
        format_features = f.readline().replace(" ", "").strip().split(",")
        for line in f: # Parse the custom features now 
            custom_features.append(line.replace(" ", "").strip())
    for format_feature in format_features:
        if vcf_header.formats[format_feature].number == 1:
            df["min_parent_{}".format(format_feature)] = df[["father_{}".format(format_feature), "mother_{}".format(format_feature)]].min(axis = 1)
            df["max_parent_{}".format(format_feature)] = df[["father_{}".format(format_feature), "mother_{}".format(format_feature)]].max(axis = 1)
        elif vcf_header.formats[format_feature].number == "G":
            for i in range(0, 3):
                df["min_parent_{}_{}".format(format_feature, str(i))] = df[["father_{}_{}".format(format_feature, str(i)), "mother_{}_{}".format(format_feature, str(i))]].min(axis = 1)
                df["max_parent_{}_{}".format(format_feature, str(i))] = df[["father_{}_{}".format(format_feature, str(i)), "mother_{}_{}".format(format_feature, str(i))]].max(axis = 1)
        elif vcf_header.formats[format_feature].number == "R":
            for i in range(0, 1):
                df["min_parent_{}_{}".format(format_feature, str(i))] = df[["father_{}_{}".format(format_feature, str(i)), "mother_{}_{}".format(format_feature, str(i))]].min(axis = 1)
                df["max_parent_{}_{}".format(format_feature, str(i))] = df[["father_{}_{}".format(format_feature, str(i)), "mother_{}_{}".format(format_feature, str(i))]].max(axis = 1)
    ## Use the function dispatch for custom features now
    #for custom_feature in custom_features:
    #    regex_match = re.match("(^\w+):(int|float)=(\w+)\(\w+\)", custom_feature)
    #    feature_name = regex_match.group(1)
    #    return_type = regex_match.group(2)
    #    function_name = regex_match.group(3)
    #    input_feature_name = regex_match.group(4)
    #    df["offspring_{}".format(feature_name)] = df[input_feature_name
    #            # To do: put this in feature_extraction code
    #            # So feature extraction code will get offspring_AR, father_AR, mother_AR
    #            # Then we'll also get min_parent_AR, max_parent_AR in the postprocessing step
    return df

"""
pedigree_dict, offspring_parents_dict, sample_sex_and_phenotype_dict = make_pedigree_dicts("tutorial.ped")
sample_index_dict, index_sample_dict = make_sample_index_dicts("tutorial.vcf.gz")
offspring_index_id_dict = make_offspring_index_dict(offspring_parents_dict, sample_index_dict)

dnm_features_dict = make_features_table("tutorial.vcf.gz", offspring_index_id_dict, offspring_parents_dict, sample_index_dict)

df_dnm_features_dict = pd.DataFrame.from_dict(dnm_features_dict).transpose().reset_index()
df_dnm_features_dict = df_dnm_features_dict.rename(columns = {"level_0": "chrom", "level_1": "pos", "level_2": "id", "level_3": "ref", "level_4": "alt", "level_5": "iid"})

df_dnm_features_dict.to_csv("df_dnm_features_dict.tsv", sep = "\t", index = False)

# Classification
clf_snv_filepath = "ssc-jg-snp-clf-half-sample.joblib"
clf_indel_filepath = "ssc-jg-indel-clf-half-sample.joblib"

# Dumb hack job for now, will change later (after I do some re-training...)
df_dnm_features_dict["ClippingRankSum"] = 0

key = ["chrom", "pos", "ref", "alt", "iid"]
format_feature_names = ["max_parental_AR", "min_parental_AR", "offspring_AR", "max_parental_DP", "min_parental_DP", "offspring_DP", "max_parental_dnm_PL", "min_parental_dnm_PL", "max_parental_hom_ref_PL", "min_parental_hom_ref_PL", "offspring_dnm_PL", "offspring_hom_ref_PL", "max_parental_GQ", "min_parental_GQ", "offspring_GQ"]
info_feature_names = ["VQSLOD", "ClippingRankSum", "BaseQRankSum", "FS", "SOR", "MQ", "MQRankSum", "QD", "ReadPosRankSum"]
key_and_features = key + format_feature_names + info_feature_names

def run_snv_classifier(df, clf_snv_filepath):
    df_snv = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)]
    df_snv_values = df_snv[key_and_features].dropna().values
    clf_snv = joblib.load(clf_snv_filepath)
    preds_snv_values = clf_snv.predict_proba(df_snv_values[:, 5:])
    df_preds  = pd.DataFrame({"chrom": df_snv_values[:, 0], "pos": df_snv_values[:, 1], "ref": df_snv_values[:, 2], "alt": df_snv_values[:, 3], "iid": df_snv_values[:, 4], "prob": preds_snv_values[:, 0]})
    # Retain the index for easy sorting later on
    df_snv["copy_index"] = df_snv.index
    return df_snv.merge(df_preds, on = key, how = "left")

def run_indel_classifier(df, clf_indel_filepath):
    df_indel = df[(df["ref"].str.len() != 1) | (df["alt"].str.len() != 1)]
    df_indel_values = df_indel[key_and_features].dropna().values
    clf_indel = joblib.load(clf_indel_filepath)
    preds_indel_values = clf_indel.predict_proba(df_indel_values[:, 5:])
    df_preds  = pd.DataFrame({"chrom": df_indel_values[:, 0], "pos": df_indel_values[:, 1], "ref": df_indel_values[:, 2], "alt": df_indel_values[:, 3], "iid": df_indel_values[:, 4], "prob": preds_indel_values[:, 0]})
    df_indel["copy_index"] = df_indel.index
    return df_indel.merge(df_preds, on = key, how = "left")

df_snv_preds = run_snv_classifier(df_dnm_features_dict, clf_snv_filepath)
df_indel_preds = run_indel_classifier(df_dnm_features_dict, clf_indel_filepath)

# Use indices for sorting
df_output = pd.concat([df_snv_preds, df_indel_preds]).set_index("copy_index").sort_index()
df_output.to_csv("df_output.tsv", sep = "\t", index = False)

# Output DNM BED
df_output["bed_name"] = df_output.apply(lambda row: row.chrom + "_" + str(row.pos) + "_" + row.ref + "_" + row.alt + "_" + row.iid, axis = 1)
df_output["start"] = df_output["pos"] - 1
df_output["end"] = df_output["pos"]
df_output_bed = df_output[["chrom", "start", "end", "bed_name", "prob"]]
dnm_pybed = pybedtools.BedTool.from_dataframe(df_output_bed)
dnm_pybed.saveas("dnms.bed")

# Output DNM VCF (or use tab-delimited table?)
# If I use tab-delimited file, then the only columns I'd add are probability for now, maybe truth label. These would be float or integer in VCF header. And we can filter DNMs based on those predictions from the input VCF.
"""
