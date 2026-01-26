#!/usr/bin/env python3
"""
Swap pedigree file for creating synthetic de novos.

Pairs up families randomly and swaps their offspring, keeping parents the same.
This creates "synthetic" parent-child relationships where inherited variants
will appear as de novos.

Usage:
    python swap_pedigree.py input.ped > output.swapped.ped
    python swap_pedigree.py input.ped -o output.swapped.ped
"""

import argparse
import sys
from random import seed, shuffle


def parse_pedigree(ped_file):
    """Parse PED file into pedigree dict and sample metadata."""
    pedigree_dict = {}
    sample_metadata = {}

    with open(ped_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split()
            if len(fields) < 6:
                continue

            fid, iid, father, mother, sex, pheno = fields[:6]

            if fid not in pedigree_dict:
                pedigree_dict[fid] = {"father_id": None, "mother_id": None, "offspring": []}

            # Parents have 0 for father/mother columns
            if father == "0" and mother == "0":
                if sex == "1":
                    pedigree_dict[fid]["father_id"] = iid
                elif sex == "2":
                    pedigree_dict[fid]["mother_id"] = iid
            else:
                pedigree_dict[fid]["offspring"].append(iid)

            sample_metadata[iid] = {"sex": sex, "pheno": pheno}

    return pedigree_dict, sample_metadata


def swap_pedigree(pedigree_dict, random_seed=0):
    """Swap offspring between pairs of families."""
    seed(random_seed)

    # Filter to complete families (have father, mother, and offspring)
    complete_families = {
        fid: data for fid, data in pedigree_dict.items()
        if data["father_id"] and data["mother_id"] and data["offspring"]
    }

    family_ids = list(complete_families.keys())
    indices = list(range(len(family_ids)))
    shuffle(indices)

    swapped = {}

    # Pair up families and swap offspring
    for i in range(0, len(indices) - 1, 2):
        fid_a = family_ids[indices[i]]
        fid_b = family_ids[indices[i + 1]]

        # Family A keeps its parents but gets Family B offspring
        swapped[fid_a] = {
            "father_id": complete_families[fid_a]["father_id"],
            "mother_id": complete_families[fid_a]["mother_id"],
            "offspring": complete_families[fid_b]["offspring"],
        }

        # Family B keeps its parents but gets Family A offspring
        swapped[fid_b] = {
            "father_id": complete_families[fid_b]["father_id"],
            "mother_id": complete_families[fid_b]["mother_id"],
            "offspring": complete_families[fid_a]["offspring"],
        }

    return swapped


def write_pedigree(swapped_dict, sample_metadata, output):
    """Write swapped pedigree in PED format."""
    for fid, data in swapped_dict.items():
        father_id = data["father_id"]
        mother_id = data["mother_id"]

        # Write offspring
        for offspring_id in data["offspring"]:
            meta = sample_metadata.get(offspring_id, {"sex": "0", "pheno": "0"})
            print(f"{fid}\t{offspring_id}\t{father_id}\t{mother_id}\t{meta['sex']}\t{meta['pheno']}", file=output)

        # Write father
        if father_id in sample_metadata:
            meta = sample_metadata[father_id]
            print(f"{fid}\t{father_id}\t0\t0\t{meta['sex']}\t{meta['pheno']}", file=output)

        # Write mother
        if mother_id in sample_metadata:
            meta = sample_metadata[mother_id]
            print(f"{fid}\t{mother_id}\t0\t0\t{meta['sex']}\t{meta['pheno']}", file=output)


def main():
    parser = argparse.ArgumentParser(description="Swap pedigree for synthetic DNM generation")
    parser.add_argument("ped_file", help="Input PED file")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed (default: 0)")
    args = parser.parse_args()

    pedigree_dict, sample_metadata = parse_pedigree(args.ped_file)
    swapped_dict = swap_pedigree(pedigree_dict, args.seed)

    output = open(args.output, "w") if args.output else sys.stdout
    write_pedigree(swapped_dict, sample_metadata, output)

    if args.output:
        output.close()
        # Print stats to stderr
        print(f"Swapped {len(swapped_dict)} families", file=sys.stderr)


if __name__ == "__main__":
    main()
