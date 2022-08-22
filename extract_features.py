import joblib
import numpy as np
import pandas as pd
import pybedtools
import pysam
import sys

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

# Extract putative de novo mutations from the VCF
def make_features_dict(vcf_filepath, offspring_index_id_dict, offspring_parents_dict, sample_index_dict):
    dnm_features_dict = {}

    header_ = "\t".join(["chrom", "pos", "id", "ref", "alt", "iid", "offspring_GT", "father_GT", "mother_GT", "nalt", "filter", "qual"])
    format_feature_names = "\t".join(["max_parental_AR", "min_parental_AR", "offspring_AR", "max_parental_DP", "min_parental_DP", "offspring_DP", "max_parental_dnm_PL", "min_parental_dnm_PL", "max_parental_hom_ref_PL", "min_parental_hom_ref_PL", "offspring_dnm_PL", "offspring_hom_ref_PL", "max_parental_GQ", "min_parental_GQ", "offspring_GQ"])
    info_feature_names = "\t".join(["VQSLOD", "ClippingRankSum", "BaseQRankSum", "FS", "SOR", "MQ", "MQRankSum", "QD", "ReadPosRankSum"])
    header = "{}\t{}\t{}".format(header_, format_feature_names, info_feature_names) 

    vcf_iterator = pysam.VariantFile(vcf_filepath, mode = "r")
    number_of_samples = len(vcf_iterator.header.samples)
    for record in vcf_iterator:
        # Skip multiallelic variants
        if len(record.alts) > 1: continue
        # Skip X and Y for now
        if "X" in record.chrom or "Y" in record.chrom: continue
        # Get INFO-level features
        # For now, these are the INFO features to use: ["VQSLOD","ClippingRankSum","BaseQRankSum","FS","SOR","MQ","MQRankSum","QD","ReadPosRankSum"]
        info_features = ["VQSLOD", "ClippingRankSum", "BaseQRankSum", "FS", "SOR", "MQ", "MQRankSum", "QD", "ReadPosRankSum"]
        dnm_info_features = {}
        for feature in info_features:
            if feature in record.info:
                dnm_info_features[feature] = record.info[feature]
            else: dnm_info_features[feature] = np.nan
        # Instead:
        # 1. Loop through offspring
        # 2. Get parents of offspring
        # 3. Get indices of offspring and parents for getting genotyping information
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
           
            # The GT FORMAT field will of course be required no matter what.
            # For now, these other FORMAT fields will be required: (AD, DP, GQ, PL)
            offspring_AD = record.samples[i]["AD"]
            father_AD = record.samples[sample_index_dict[father_id]]["AD"]
            mother_AD = record.samples[sample_index_dict[mother_id]]["AD"]

            offspring_DP = record.samples[i]["DP"]
            father_DP = record.samples[sample_index_dict[father_id]]["DP"]
            mother_DP = record.samples[sample_index_dict[mother_id]]["DP"]

            offspring_GQ = record.samples[i]["GQ"]
            father_GQ = record.samples[sample_index_dict[father_id]]["GQ"]
            mother_GQ = record.samples[sample_index_dict[mother_id]]["GQ"]

            # The PLs are like so: (homozygous_reference_PL, heterozygous_PL, homozygous_alternate_PL) 
            # For the offspring, heterozygous_PL should always be 0 (since that corresponds to the genotype)
            # For the parents, homozygous_reference_PL should always be 0 (again, that corresponds to their genotypes)
            offspring_PL = record.samples[i]["PL"]
            father_PL = record.samples[sample_index_dict[father_id]]["PL"]
            mother_PL = record.samples[sample_index_dict[mother_id]]["PL"]

            # Getting post-processed features
            buffer_ = 1.0

            offspring_AR = float(offspring_AD[1]) / (float(offspring_AD[0]) + buffer_)
            if offspring_AR > 1.0: offspring_AR = 1.0 / offspring_AR # Should always be a fraction?
            father_AR = float(father_AD[1]) / (float(father_AD[0]) + buffer_)
            if father_AR > 1.0: father_AR = 1.0 / father_AR # Should always be a fraction?
            mother_AR = float(mother_AD[1]) / (float(mother_AD[0]) + buffer_)
            if mother_AR > 1.0: mother_AR = 1.0 / mother_AR # Should always be a fraction?

            offspring_AD_log2_coverage_ratio = np.log2( (offspring_AD[0] + offspring_AD[1] + buffer_) / (np.median([offspring_AD[0], offspring_AD[1]]) + buffer_) )
            father_AD_log2_coverage_ratio = np.log2( (father_AD[0] + father_AD[1] + buffer_) / (np.median([father_AD[0], father_AD[1]]) + buffer_) )
            mother_AD_log2_coverage_ratio = np.log2( (mother_AD[0] + mother_AD[1] + buffer_) / (np.median([mother_AD[0], mother_AD[1]]) + buffer_) )

            offspring_dnm_PL = offspring_PL[1]
            offspring_hom_ref_PL = offspring_PL[0]

            (min_parental_AR, max_parental_AR) = sorted([father_AR, mother_AR])
            (min_parental_DP, max_parental_DP) = sorted([father_AD_log2_coverage_ratio, mother_AD_log2_coverage_ratio])
            (min_parental_GQ, max_parental_GQ) = sorted([father_GQ, mother_GQ])
            (min_parental_dnm_PL, max_parental_dnm_PL) = sorted([father_PL[1], mother_PL[1]])
            (min_parental_hom_ref_PL, max_parental_hom_ref_PL) = sorted([father_PL[0], mother_PL[0]]) # These should both be 0...

            key = (record.chrom, record.pos, record.id, record.ref, record.alts[0], offspring_id)
            dnm_features_dict[key] = {}

            dnm_features_dict[key]["offspring_GT"] = offspring_GT
            dnm_features_dict[key]["father_GT"] = father_GT 
            dnm_features_dict[key]["mother_GT"] = mother_GT 

            # Placeholder values (and they're not used anyway)
            dnm_features_dict[key]["n_alt"] = "NA"
            dnm_features_dict[key]["filter"] = "NA"
            dnm_features_dict[key]["qual"] = "NA"

            # FORMAT-level features

            dnm_features_dict[key]["max_parental_AR"] = max_parental_AR 
            dnm_features_dict[key]["min_parental_AR"] = min_parental_AR 
            dnm_features_dict[key]["offspring_AR"] = offspring_AR
            dnm_features_dict[key]["max_parental_DP"] = max_parental_DP
            dnm_features_dict[key]["min_parental_DP"] = min_parental_DP
            dnm_features_dict[key]["offspring_DP"] = offspring_DP
            dnm_features_dict[key]["max_parental_dnm_PL"] = max_parental_dnm_PL
            dnm_features_dict[key]["min_parental_dnm_PL"] = min_parental_dnm_PL
            dnm_features_dict[key]["max_parental_hom_ref_PL"] = max_parental_hom_ref_PL
            dnm_features_dict[key]["min_parental_hom_ref_PL"] = min_parental_hom_ref_PL
            dnm_features_dict[key]["offspring_dnm_PL"] = offspring_dnm_PL
            dnm_features_dict[key]["offspring_hom_ref_PL"] = offspring_hom_ref_PL
            dnm_features_dict[key]["max_parental_GQ"] = max_parental_GQ
            dnm_features_dict[key]["min_parental_GQ"] = min_parental_GQ
            dnm_features_dict[key]["offspring_GQ"] = offspring_GQ

            # Add INFO-level features
            # dnm_features_dict[key] |= dnm_info_features
            dnm_features_dict[key].update(dnm_info_features)

    return dnm_features_dict

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
