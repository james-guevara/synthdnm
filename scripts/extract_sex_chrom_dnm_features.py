#!/usr/bin/env python3
"""
Extract features for sex chromosome DNM candidates.

Two modes:
1. Putative DNMs (default):
   - Child has alt (HET or HOM_ALT), both parents don't (HOM_REF or MISSING)
   - Output suffix: _real.tsv (matches autosomal convention)

2. Synthetic DNMs (--swapped_ped):
   - Child has alt, at least one real parent has alt (inherited)
   - Both swapped parents don't have alt (HOM_REF or MISSING)
   - Output suffix: _synthetic.tsv

Naive extraction — caller-specific filtering (e.g., male het on non-PAR chrX
= artifact for GATK) is handled downstream in preprocessing.

Differences from extract_dnm_features.py (autosomal):
  - GT matching: child HET or HOM_ALT, parents HOM_REF or MISSING
  - AC: configurable via --max_ac (default: 4)
  - Adds child_gt_type, father_gt_type, mother_gt_type columns (cyvcf2
    gt_types: 0=HOM_REF, 1=HET, 2=UNKNOWN, 3=HOM_ALT) for downstream
    caller-specific filtering

Output format matches extract_dnm_features.py (same columns + gt_type columns)
so preprocessing can handle all features uniformly.

Usage:
    # Putative DNMs
    python extract_sex_chrom_dnm_features.py \\
        --vcf input.vcf.gz \\
        --ped real.psam \\
        --region chrX:1-50000000 \\
        --output chrX_0-50000000_real.tsv \\
        --gatk

    # Synthetic DNMs
    python extract_sex_chrom_dnm_features.py \\
        --vcf input.vcf.gz \\
        --ped real.psam \\
        --swapped_ped swapped.psam \\
        --region chrX:1-50000000 \\
        --output chrX_0-50000000_synthetic.tsv \\
        --gatk
"""

import argparse
import sys

import numpy as np
from cyvcf2 import VCF


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract features for sex chromosome DNM candidates."
    )
    parser.add_argument("--vcf", required=True, help="Path to input VCF file")
    parser.add_argument("--ped", required=True, help="Path to pedigree file (.psam/.fam)")
    parser.add_argument("--swapped_ped",
                        help="Path to swapped pedigree file (enables synthetic DNM mode)")
    parser.add_argument("--region", help="Region to query (e.g., chrX:1-50000000)")
    parser.add_argument("--output", help="Output TSV file (default: stdout)")
    parser.add_argument("--max_ac", type=int, default=4,
                        help="Maximum allele count to include (default: 4)")
    parser.add_argument("--gatk", action="store_true",
                        help="Extract GATK-specific INFO fields")
    return parser.parse_args()


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
            iid, father, mother, sex = fields[1], fields[2], fields[3], fields[4]
            if father == "0" and mother == "0":
                continue
            trios[iid] = (father, mother, sex)
    return trios


def build_trio_arrays(trios, sample_to_idx):
    """Build numpy arrays for vectorized trio checking."""
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
        child_sex,
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
    """Extract all FORMAT fields once per variant."""
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


def get_info_fields(variant, alt_index, args):
    """Extract INFO fields once per variant."""
    info = {}
    if args.gatk:
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
    base_cols = [
        "#CHROM", "POS0", "POS", "REF", "ALT", "ALT_specific",
        "allele_num", "SAMPLE", "AC",
    ]
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
    feature_cols.extend([
        "indel_flag", "haploid_flag",
        "child_gt_type", "father_gt_type", "mother_gt_type",
    ])
    return base_cols + feature_cols


def has_alt(gt_type):
    """Check if gt_type indicates the sample carries an alt allele."""
    return (gt_type == 1) | (gt_type == 3)  # HET or HOM_ALT


def no_alt(gt_type):
    """Check if gt_type indicates the sample does not carry an alt allele."""
    return (gt_type == 0) | (gt_type == 2)  # HOM_REF or UNKNOWN/MISSING


def write_row(variant, alt_index, allele_ac, fmt, child_idx, father_idx, mother_idx,
              child_name, child_sex_val, child_gt_val, father_gt_val, mother_gt_val,
              args, out_file):
    """Write a single output row."""
    alt_allele = variant.ALT[alt_index - 1]
    alt_full = ",".join(variant.ALT)
    info = get_info_fields(variant, alt_index, args)

    cf = extract_sample_features(fmt, child_idx)
    ff = extract_sample_features(fmt, father_idx)
    mf = extract_sample_features(fmt, mother_idx)

    indel_flag = 1 if len(variant.REF) != len(alt_allele) else 0
    haploid_flag = 1 if child_sex_val == "1" else 0

    row = [
        variant.CHROM,
        str(variant.POS - 1),
        str(variant.POS),
        variant.REF,
        alt_full,
        alt_allele,
        str(alt_index),
        child_name,
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

    row.extend([
        str(indel_flag),
        str(haploid_flag),
        str(child_gt_val),
        str(father_gt_val),
        str(mother_gt_val),
    ])
    print("\t".join(row), file=out_file)


def process_putative_dnms(variant, alt_index, allele_ac, gt_types,
                           child_indices, father_indices, mother_indices,
                           child_names, child_sex, args, out_file):
    """Extract putative DNM candidates on sex chromosomes.

    Child has alt, both parents don't have alt (or are missing).
    """
    child_gt = gt_types[child_indices]
    father_gt = gt_types[father_indices]
    mother_gt = gt_types[mother_indices]

    # At least one parent must be explicitly HOM_REF — if both are missing,
    # we can't determine if it's truly de novo
    at_least_one_ref = (father_gt == 0) | (mother_gt == 0)
    valid_mask = has_alt(child_gt) & no_alt(father_gt) & no_alt(mother_gt) & at_least_one_ref
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return 0

    alt_allele = variant.ALT[alt_index - 1]
    if alt_allele == "*":
        return 0

    fmt = extract_format_fields(variant)

    count = 0
    for idx in valid_indices:
        write_row(variant, alt_index, allele_ac, fmt,
                  child_indices[idx], father_indices[idx], mother_indices[idx],
                  child_names[idx], child_sex[idx],
                  child_gt[idx], father_gt[idx], mother_gt[idx],
                  args, out_file)
        count += 1

    return count


def process_synthetic_dnms(variant, alt_index, allele_ac, gt_types,
                            real_child_indices, real_father_indices, real_mother_indices,
                            swap_child_indices, swap_father_indices, swap_mother_indices,
                            child_names, child_sex, child_to_real_idx, child_to_swap_idx,
                            args, out_file):
    """Extract synthetic DNM candidates on sex chromosomes.

    Child has alt, at least one real parent has alt (inherited),
    both swapped parents don't have alt.

    Naive: doesn't check which parent should biologically carry the variant
    based on sex/chromosome — that filtering happens in preprocessing.
    """
    alt_allele = variant.ALT[alt_index - 1]
    if alt_allele == "*":
        return 0

    fmt = extract_format_fields(variant)
    count = 0

    # Find all samples that have alt
    samples_with_alt = np.where(has_alt(gt_types))[0]

    for child_vcf_idx in samples_with_alt:
        if child_vcf_idx not in child_to_real_idx:
            continue

        real_trio_idx = child_to_real_idx[child_vcf_idx]
        real_father_idx = real_father_indices[real_trio_idx]
        real_mother_idx = real_mother_indices[real_trio_idx]

        # At least one real parent must have alt (inherited)
        father_has = has_alt(np.array([gt_types[real_father_idx]]))[0]
        mother_has = has_alt(np.array([gt_types[real_mother_idx]]))[0]
        if not father_has and not mother_has:
            continue

        # Check swapped pedigree
        if child_vcf_idx not in child_to_swap_idx:
            continue

        swap_trio_idx = child_to_swap_idx[child_vcf_idx]
        swap_father_idx = swap_father_indices[swap_trio_idx]
        swap_mother_idx = swap_mother_indices[swap_trio_idx]

        # Both swapped parents must not have alt, and at least one must be
        # explicitly HOM_REF (not both missing)
        swap_f_gt = gt_types[swap_father_idx]
        swap_m_gt = gt_types[swap_mother_idx]
        swap_f_no = no_alt(np.array([swap_f_gt]))[0]
        swap_m_no = no_alt(np.array([swap_m_gt]))[0]
        if not swap_f_no or not swap_m_no:
            continue
        if swap_f_gt != 0 and swap_m_gt != 0:
            continue

        # Valid synthetic DNM — extract features using swapped parents
        child_gt_val = gt_types[child_vcf_idx]
        swap_f_gt_val = gt_types[swap_father_idx]
        swap_m_gt_val = gt_types[swap_mother_idx]

        write_row(variant, alt_index, allele_ac, fmt,
                  child_vcf_idx, swap_father_idx, swap_mother_idx,
                  child_names[real_trio_idx], child_sex[real_trio_idx],
                  child_gt_val, swap_f_gt_val, swap_m_gt_val,
                  args, out_file)
        count += 1

    return count


def main():
    args = parse_args()
    synthetic_mode = args.swapped_ped is not None

    # Load pedigrees
    real_trios = load_pedigree(args.ped)
    swapped_trios = load_pedigree(args.swapped_ped) if synthetic_mode else None

    # Open VCF
    vcf = VCF(args.vcf, strict_gt=True)
    sample_to_idx = {s: i for i, s in enumerate(vcf.samples)}

    # Build vectorized trio arrays
    real_child_idx, real_father_idx, real_mother_idx, child_names, child_sex = \
        build_trio_arrays(real_trios, sample_to_idx)

    # Build child index lookups
    child_to_real_idx = {real_child_idx[i]: i for i in range(len(real_child_idx))}

    if synthetic_mode:
        swap_child_idx, swap_father_idx, swap_mother_idx, _, _ = \
            build_trio_arrays(swapped_trios, sample_to_idx)
        child_to_swap_idx = {swap_child_idx[i]: i for i in range(len(swap_child_idx))}
    else:
        swap_child_idx = swap_father_idx = swap_mother_idx = None
        child_to_swap_idx = {}

    mode_str = "synthetic" if synthetic_mode else "putative"
    print(f"Mode: {mode_str} DNMs", file=sys.stderr)
    print(f"Loaded {len(child_names)} trios", file=sys.stderr)
    print(f"AC filter: max_ac={args.max_ac}", file=sys.stderr)

    # Output setup
    out_file = open(args.output, "w") if args.output else sys.stdout
    header = get_header(args)
    print("\t".join(header), file=out_file)

    # Track counts
    n_variants = 0
    n_features = 0
    n_skipped_ac = 0

    # Iterate through variants
    region = args.region if args.region else None
    for variant in vcf(region):
        n_variants += 1
        if n_variants % 100000 == 0:
            print(f"Processed {n_variants:,} variants, "
                  f"extracted {n_features:,} features "
                  f"(skipped {n_skipped_ac:,} AC-filtered)",
                  file=sys.stderr)

        if len(variant.ALT) == 0:
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

            if allele_ac > args.max_ac:
                n_skipped_ac += 1
                continue

            if synthetic_mode:
                n_features += process_synthetic_dnms(
                    variant, alt_idx, allele_ac, gt_types,
                    real_child_idx, real_father_idx, real_mother_idx,
                    swap_child_idx, swap_father_idx, swap_mother_idx,
                    child_names, child_sex, child_to_real_idx, child_to_swap_idx,
                    args, out_file,
                )
            else:
                n_features += process_putative_dnms(
                    variant, alt_idx, allele_ac, gt_types,
                    real_child_idx, real_father_idx, real_mother_idx,
                    child_names, child_sex, args, out_file,
                )

    print(f"\nDone. Processed {n_variants:,} variants, "
          f"extracted {n_features:,} {mode_str} features", file=sys.stderr)
    if n_skipped_ac > 0:
        print(f"  Skipped {n_skipped_ac:,} alleles with AC > {args.max_ac}",
              file=sys.stderr)

    if args.output:
        out_file.close()


if __name__ == "__main__":
    main()
