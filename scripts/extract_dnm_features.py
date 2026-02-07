#!/usr/bin/env python3
"""
Extract features for DNM classification.

Two modes:
1. Synthetic DNMs (positive training examples):
   - Filter to AC=2, biallelic variants
   - Use swapped pedigree: child 0/1, swapped parents 0/0
   - Validate with real pedigree: one real parent must be 0/1 (inherited)

2. Real putative DNMs (negative training examples):
   - No AC filter - consider all variants
   - Use real pedigree: child 0/1, both parents 0/0

Features extracted are raw "protofeatures" - calculations (e.g., allele ratio)
should be done downstream.

Usage:
    # Synthetic DNMs
    python extract_dnm_features.py \\
        --vcf input.vcf.gz \\
        --ped real.psam \\
        --swapped_ped swapped.psam \\
        --region chr1:1-50000000 \\
        --output chr1_synthetic.tsv

    # Real putative DNMs
    python extract_dnm_features.py \\
        --vcf input.vcf.gz \\
        --ped real.psam \\
        --real_dnms \\
        --region chr1:1-50000000 \\
        --output chr1_real.tsv
"""

import argparse
import sys
from cyvcf2 import VCF
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract features for DNM classification."
    )
    parser.add_argument("--vcf", required=True, help="Path to input VCF file")
    parser.add_argument("--ped", required=True, help="Path to real pedigree file")
    parser.add_argument("--swapped_ped", help="Path to swapped pedigree file (required for synthetic DNMs)")
    parser.add_argument("--region", help="Region to query (e.g., chr1:1-50000000)")
    parser.add_argument("--output", help="Output TSV file (default: stdout)")
    parser.add_argument("--real_dnms", action="store_true",
                        help="Extract real putative DNMs (no AC filter)")
    parser.add_argument("--gatk", action="store_true",
                        help="Extract GATK-specific INFO fields")
    args = parser.parse_args()

    if not args.real_dnms and not args.swapped_ped:
        parser.error("--swapped_ped is required unless --real_dnms is specified")

    return args


def load_pedigree(ped_path):
    """Load pedigree file and return dict of child -> (father, mother, sex)."""
    trios = {}
    with open(ped_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 6:
                fields = line.strip().split()
            if len(fields) < 6:
                continue
            fid, iid, father, mother, sex = fields[0], fields[1], fields[2], fields[3], fields[4]
            if father == "0" and mother == "0":
                continue
            trios[iid] = (father, mother, sex)
    return trios


def build_trio_arrays(trios, sample_to_idx):
    """Build numpy arrays for vectorized trio checking.

    Returns:
        child_indices: array of child sample indices
        father_indices: array of father sample indices
        mother_indices: array of mother sample indices
        child_names: list of child sample names (for output)
        child_sex: list of child sex values
    """
    child_indices = []
    father_indices = []
    mother_indices = []
    child_names = []
    child_sex = []

    for child, (father, mother, sex) in trios.items():
        if child in sample_to_idx and father in sample_to_idx and mother in sample_to_idx:
            child_indices.append(sample_to_idx[child])
            father_indices.append(sample_to_idx[father])
            mother_indices.append(sample_to_idx[mother])
            child_names.append(child)
            child_sex.append(sex)

    return (
        np.array(child_indices, dtype=np.int32),
        np.array(father_indices, dtype=np.int32),
        np.array(mother_indices, dtype=np.int32),
        child_names,
        child_sex
    )


def format_array(arr):
    """Format a numpy array as comma-separated string."""
    if arr is None:
        return "."
    return ",".join(str(x) for x in arr)


def get_fmt_scalar(fmt_array, sample_idx):
    """Get a scalar FORMAT value for a sample, returning '.' if missing."""
    if fmt_array is None:
        return "."
    val = fmt_array[sample_idx][0]
    # cyvcf2 uses min int/float as missing sentinel
    if isinstance(val, (np.integer, int)):
        if val == -2147483648:
            return "."
        return int(val)
    if isinstance(val, (np.floating, float)):
        if np.isnan(val) or val < -1e37:
            return "."
        return val
    return val


def get_fmt_string(fmt_array, sample_idx):
    """Get a string FORMAT value for a sample, returning '.' if missing."""
    if fmt_array is None:
        return "."
    val = fmt_array[sample_idx]
    if isinstance(val, bytes):
        val = val.decode()
    if val is None or val == "" or val == ".":
        return "."
    return val


def extract_format_fields(variant):
    """Extract all FORMAT fields once per variant. Returns dict of field arrays.

    All fields use try/except to handle VCFs that don't define certain fields
    in the header. Missing fields get None, which downstream functions render as '.'.
    """
    fmt = {}
    for field in ("AD", "GQ", "PL", "DP", "AB", "FT", "PGT", "PID"):
        try:
            fmt[field] = variant.format(field)
        except Exception:
            fmt[field] = None
    return fmt


def extract_sample_features(fmt, sample_idx):
    """Extract all FORMAT features for a single sample."""
    f = {}
    f["AD"] = format_array(fmt["AD"][sample_idx]) if fmt["AD"] is not None else "."
    f["GQ"] = get_fmt_scalar(fmt["GQ"], sample_idx)
    f["PL"] = format_array(fmt["PL"][sample_idx]) if fmt["PL"] is not None else "."
    f["DP"] = get_fmt_scalar(fmt["DP"], sample_idx)
    f["AB"] = get_fmt_scalar(fmt["AB"], sample_idx)
    f["FT"] = get_fmt_string(fmt["FT"], sample_idx)
    f["PGT"] = get_fmt_string(fmt["PGT"], sample_idx)
    f["PID"] = get_fmt_string(fmt["PID"], sample_idx)
    return f


def extract_features_batch(variant, valid_indices, child_indices, father_indices,
                           mother_indices, args):
    """Extract features for multiple valid DNM candidates at once."""
    features_list = []

    # Get all format fields once
    fmt = extract_format_fields(variant)

    for i in valid_indices:
        child_idx = child_indices[i]
        father_idx = father_indices[i]
        mother_idx = mother_indices[i]

        features = {
            "child": extract_sample_features(fmt, child_idx),
            "father": extract_sample_features(fmt, father_idx),
            "mother": extract_sample_features(fmt, mother_idx),
        }
        features_list.append(features)

    return features_list


def get_info_fields(variant, alt_index, args):
    """Extract INFO fields once per variant."""
    info = {}
    if args.gatk:
        # Core GATK INFO fields
        info["INFO_DP"] = variant.INFO.get("DP", ".")
        info["INFO_ExcessHet"] = variant.INFO.get("ExcessHet", ".")
        info["INFO_FS"] = variant.INFO.get("FS", ".")
        info["INFO_MQ"] = variant.INFO.get("MQ", ".")
        info["INFO_QD"] = variant.INFO.get("QD", ".")
        info["INFO_SOR"] = variant.INFO.get("SOR", ".")
        info["INFO_VQSLOD"] = variant.INFO.get("VQSLOD", ".")
        info["INFO_BaseQRankSum"] = variant.INFO.get("BaseQRankSum", ".")
        info["INFO_MQRankSum"] = variant.INFO.get("MQRankSum", ".")
        info["INFO_ReadPosRankSum"] = variant.INFO.get("ReadPosRankSum", ".")
        # Additional INFO fields
        info["INFO_AF"] = variant.INFO.get("AF", ".")
        info["INFO_AN"] = variant.INFO.get("AN", ".")
        info["INFO_InbreedingCoeff"] = variant.INFO.get("InbreedingCoeff", ".")
        info["INFO_ClippingRankSum"] = variant.INFO.get("ClippingRankSum", ".")
        info["INFO_HaplotypeScore"] = variant.INFO.get("HaplotypeScore", ".")
        info["INFO_MLEAC"] = variant.INFO.get("MLEAC", ".")
        info["INFO_MLEAF"] = variant.INFO.get("MLEAF", ".")
        info["INFO_RAW_MQ"] = variant.INFO.get("RAW_MQ", ".")
        info["INFO_MQ0"] = variant.INFO.get("MQ0", ".")
        info["INFO_DS"] = 1 if variant.INFO.get("DS") is not None else 0
        info["INFO_NEGATIVE_TRAIN_SITE"] = 1 if variant.INFO.get("NEGATIVE_TRAIN_SITE") is not None else 0
        info["INFO_POSITIVE_TRAIN_SITE"] = 1 if variant.INFO.get("POSITIVE_TRAIN_SITE") is not None else 0
        info["INFO_culprit"] = variant.INFO.get("culprit", ".")
        info["INFO_VariantType"] = variant.INFO.get("VariantType", ".")
    else:
        aq = variant.INFO.get("AQ")
        if aq is not None:
            if isinstance(aq, tuple) and len(variant.ALT) > 1:
                info["INFO_AQ"] = aq[alt_index - 1]
            else:
                info["INFO_AQ"] = aq
        else:
            info["INFO_AQ"] = "."
    return info


def get_header(args):
    """Return TSV header."""
    base_cols = ["#CHROM", "POS0", "POS", "REF", "ALT", "ALT_specific", "allele_num", "SAMPLE", "AC"]
    feature_cols = [
        "child_AD", "father_AD", "mother_AD",
        "child_GQ", "father_GQ", "mother_GQ",
        "child_PL", "father_PL", "mother_PL",
        "child_DP", "father_DP", "mother_DP",
        "child_AB", "father_AB", "mother_AB",
        "child_FT", "father_FT", "mother_FT",
        "child_PGT", "father_PGT", "mother_PGT",
        "child_PID", "father_PID", "mother_PID",
    ]
    if args.gatk:
        feature_cols.extend([
            "INFO_DP", "INFO_ExcessHet", "INFO_FS", "INFO_MQ", "INFO_QD",
            "INFO_SOR", "INFO_VQSLOD", "INFO_BaseQRankSum", "INFO_MQRankSum",
            "INFO_ReadPosRankSum",
            "INFO_AF", "INFO_AN", "INFO_InbreedingCoeff",
            "INFO_ClippingRankSum", "INFO_HaplotypeScore",
            "INFO_MLEAC", "INFO_MLEAF", "INFO_RAW_MQ", "INFO_MQ0",
            "INFO_DS", "INFO_NEGATIVE_TRAIN_SITE", "INFO_POSITIVE_TRAIN_SITE",
            "INFO_culprit", "INFO_VariantType",
        ])
    else:
        feature_cols.append("INFO_AQ")
    feature_cols.extend(["indel_flag", "haploid_flag"])
    return base_cols + feature_cols


def process_real_dnm_vectorized(variant, alt_index, allele_ac, gt_types,
                                 child_indices, father_indices, mother_indices,
                                 child_names, child_sex, args, out_file):
    """Process variant for real putative DNM extraction using vectorized operations."""

    # Vectorized genotype checking - all trios at once
    child_gt = gt_types[child_indices]
    father_gt = gt_types[father_indices]
    mother_gt = gt_types[mother_indices]

    # Valid DNM: child is het (1), both parents are hom ref (0)
    valid_mask = (child_gt == 1) & (father_gt == 0) & (mother_gt == 0)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return 0

    # Get ALT info
    alt_allele = variant.ALT[alt_index - 1]
    if alt_allele == "*":
        return 0

    alt_full = ",".join(variant.ALT)

    # Get INFO fields once
    info = get_info_fields(variant, alt_index, args)

    # Extract features for valid candidates
    features_list = extract_features_batch(
        variant, valid_indices, child_indices, father_indices, mother_indices, args
    )

    # Write output for each valid candidate
    count = 0
    for i, feat_idx in enumerate(valid_indices):
        child = child_names[feat_idx]
        sex = child_sex[feat_idx]
        features = features_list[i]

        # Indel flag
        indel_flag = 1 if len(variant.REF) != len(alt_allele) else 0

        # Haploid flag
        haploid_flag = 1 if sex == "1" and ("X" in variant.CHROM or "Y" in variant.CHROM) else 0

        # Build row
        cf = features["child"]
        ff = features["father"]
        mf = features["mother"]
        row = [
            variant.CHROM,
            str(variant.POS - 1),
            str(variant.POS),
            variant.REF,
            alt_full,
            alt_allele,
            str(alt_index),
            child,
            str(allele_ac),
            str(cf["AD"]), str(ff["AD"]), str(mf["AD"]),
            str(cf["GQ"]), str(ff["GQ"]), str(mf["GQ"]),
            str(cf["PL"]), str(ff["PL"]), str(mf["PL"]),
            str(cf["DP"]), str(ff["DP"]), str(mf["DP"]),
            str(cf["AB"]), str(ff["AB"]), str(mf["AB"]),
            str(cf["FT"]), str(ff["FT"]), str(mf["FT"]),
            str(cf["PGT"]), str(ff["PGT"]), str(mf["PGT"]),
            str(cf["PID"]), str(ff["PID"]), str(mf["PID"]),
        ]

        for key in info:
            row.append(str(info[key]))

        row.extend([str(indel_flag), str(haploid_flag)])
        print("\t".join(row), file=out_file)
        count += 1

    return count


def process_synthetic_dnm_vectorized(variant, alt_index, allele_ac, gt_types,
                                      real_child_indices, real_father_indices, real_mother_indices,
                                      swap_child_indices, swap_father_indices, swap_mother_indices,
                                      child_names, child_sex, child_to_real_idx, child_to_swap_idx,
                                      args, out_file):
    """Process variant for synthetic DNM extraction using vectorized operations."""

    # AC=2 filter
    if allele_ac != 2:
        return 0

    # Check for exactly 2 hets and no hom_alt
    het_indices = np.where(gt_types == 1)[0]
    if len(het_indices) != 2:
        return 0
    if np.any(gt_types == 3):  # hom_alt
        return 0

    # Get ALT info
    alt_allele = variant.ALT[alt_index - 1]
    if alt_allele == "*":
        return 0

    alt_full = ",".join(variant.ALT)
    info = get_info_fields(variant, alt_index, args)

    count = 0

    # Get format fields once before the loop
    fmt = extract_format_fields(variant)

    # Check each het to see if it's a valid synthetic DNM
    for child_idx in het_indices:
        if child_idx not in child_to_real_idx:
            continue

        real_trio_idx = child_to_real_idx[child_idx]
        real_father_idx = real_father_indices[real_trio_idx]
        real_mother_idx = real_mother_indices[real_trio_idx]

        # Other het must be one of the real parents
        other_het = [h for h in het_indices if h != child_idx][0]
        if other_het != real_father_idx and other_het != real_mother_idx:
            continue

        # Check swapped pedigree
        if child_idx not in child_to_swap_idx:
            continue

        swap_trio_idx = child_to_swap_idx[child_idx]
        swap_father_idx = swap_father_indices[swap_trio_idx]
        swap_mother_idx = swap_mother_indices[swap_trio_idx]

        # Both swapped parents must be hom ref
        if gt_types[swap_father_idx] != 0 or gt_types[swap_mother_idx] != 0:
            continue

        # Valid synthetic DNM - extract features using swapped parents
        child = child_names[real_trio_idx]
        sex = child_sex[real_trio_idx]

        cf = extract_sample_features(fmt, child_idx)
        ff = extract_sample_features(fmt, swap_father_idx)
        mf = extract_sample_features(fmt, swap_mother_idx)

        indel_flag = 1 if len(variant.REF) != len(alt_allele) else 0
        haploid_flag = 1 if sex == "1" and ("X" in variant.CHROM or "Y" in variant.CHROM) else 0

        row = [
            variant.CHROM, str(variant.POS - 1), str(variant.POS), variant.REF,
            alt_full, alt_allele, str(alt_index), child, str(allele_ac),
            str(cf["AD"]), str(ff["AD"]), str(mf["AD"]),
            str(cf["GQ"]), str(ff["GQ"]), str(mf["GQ"]),
            str(cf["PL"]), str(ff["PL"]), str(mf["PL"]),
            str(cf["DP"]), str(ff["DP"]), str(mf["DP"]),
            str(cf["AB"]), str(ff["AB"]), str(mf["AB"]),
            str(cf["FT"]), str(ff["FT"]), str(mf["FT"]),
            str(cf["PGT"]), str(ff["PGT"]), str(mf["PGT"]),
            str(cf["PID"]), str(ff["PID"]), str(mf["PID"]),
        ]

        for key in info:
            row.append(str(info[key]))

        row.extend([str(indel_flag), str(haploid_flag)])
        print("\t".join(row), file=out_file)
        count += 1

    return count


def main():
    args = parse_args()

    # Load pedigrees
    real_trios = load_pedigree(args.ped)
    swapped_trios = load_pedigree(args.swapped_ped) if args.swapped_ped else None

    # Open VCF
    vcf = VCF(args.vcf, strict_gt=True)
    sample_to_idx = {s: i for i, s in enumerate(vcf.samples)}

    # Build vectorized trio arrays
    real_child_idx, real_father_idx, real_mother_idx, child_names, child_sex = \
        build_trio_arrays(real_trios, sample_to_idx)

    # Build child index lookup for synthetic DNMs
    child_to_real_idx = {real_child_idx[i]: i for i in range(len(real_child_idx))}

    if swapped_trios:
        swap_child_idx, swap_father_idx, swap_mother_idx, _, _ = \
            build_trio_arrays(swapped_trios, sample_to_idx)
        child_to_swap_idx = {swap_child_idx[i]: i for i in range(len(swap_child_idx))}
    else:
        swap_child_idx = swap_father_idx = swap_mother_idx = None
        child_to_swap_idx = {}

    print(f"Loaded {len(child_names)} trios for vectorized processing", file=sys.stderr)

    # Output setup
    out_file = open(args.output, "w") if args.output else sys.stdout
    header = get_header(args)
    print("\t".join(header), file=out_file)

    # Track counts
    n_variants = 0
    n_features = 0

    # Iterate through variants
    region = args.region if args.region else None
    for variant in vcf(region):
        n_variants += 1
        if n_variants % 100000 == 0:
            print(f"Processed {n_variants} variants, extracted {n_features} features", file=sys.stderr)

        if len(variant.ALT) == 0:
            continue

        if not args.real_dnms and len(variant.ALT) > 2:
            continue

        if "*" in variant.ALT:
            continue

        gt_types = variant.gt_types

        ac = variant.INFO.get("AC")
        if ac is None:
            ac = tuple([1] * len(variant.ALT))
        elif not isinstance(ac, tuple):
            ac = (ac,)

        for alt_idx, allele_ac in enumerate(ac, start=1):
            if alt_idx > len(variant.ALT):
                break

            if args.real_dnms:
                n_features += process_real_dnm_vectorized(
                    variant, alt_idx, allele_ac, gt_types,
                    real_child_idx, real_father_idx, real_mother_idx,
                    child_names, child_sex, args, out_file
                )
            else:
                n_features += process_synthetic_dnm_vectorized(
                    variant, alt_idx, allele_ac, gt_types,
                    real_child_idx, real_father_idx, real_mother_idx,
                    swap_child_idx, swap_father_idx, swap_mother_idx,
                    child_names, child_sex, child_to_real_idx, child_to_swap_idx,
                    args, out_file
                )

    print(f"Done. Processed {n_variants} variants, extracted {n_features} features", file=sys.stderr)

    if args.output:
        out_file.close()


if __name__ == "__main__":
    main()
