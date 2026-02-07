#!/usr/bin/env python3
"""
Generate genomic regions for sharded processing.

Takes chromosome lengths from either:
1. VCF file (uses cyvcf2 to read contig info from header)
2. FASTA index file (.fai)

Outputs regions of specified size for parallel processing.

Usage:
    # From VCF
    python generate_regions.py --vcf input.vcf.gz --region-size 50000000 > regions.txt

    # From FASTA index
    python generate_regions.py --fai reference.fa.fai --region-size 50000000 > regions.txt

    # Specify chromosomes to include
    python generate_regions.py --fai ref.fa.fai --chroms chr1,chr2,chr22 --region-size 50000000

    # Output as BED format (0-based)
    python generate_regions.py --fai ref.fa.fai --format bed > regions.bed

Output formats:
    region (default, 1-based):
        chr1:1-50000000
        chr1:50000001-100000000

    bed (0-based, tab-separated):
        chr1	0	50000000
        chr1	50000000	100000000
"""

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate genomic regions for sharded processing."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--vcf", help="VCF file (uses cyvcf2 to read contig info)")
    input_group.add_argument("--fai", help="FASTA index file (.fai)")

    parser.add_argument("--region-size", type=int, default=50_000_000,
                        help="Region size in bp (default: 50000000 = 50MB)")
    parser.add_argument("--chroms", help="Comma-separated list of chromosomes to include (default: chr1-22,X,Y)")
    parser.add_argument("--format", choices=["region", "bed"], default="region",
                        help="Output format: 'region' (chr:start-end, 1-based) or 'bed' (tab-separated, 0-based)")
    parser.add_argument("--output", help="Output file (default: stdout)")

    return parser.parse_args()


def get_default_chroms():
    """Return default list of chromosomes (chr1-22, chrX, chrY)."""
    chroms = [f"chr{i}" for i in range(1, 23)]
    chroms.extend(["chrX", "chrY"])
    return set(chroms)


def parse_vcf_contigs(vcf_path):
    """Use cyvcf2 to get chromosome lengths from VCF header."""
    from cyvcf2 import VCF

    chrom_lengths = {}
    vcf = VCF(vcf_path)

    # cyvcf2 provides seqlens and seqnames from the header
    for name, length in zip(vcf.seqnames, vcf.seqlens):
        chrom_lengths[name] = length

    vcf.close()
    return chrom_lengths


def parse_fai(fai_path):
    """Parse FASTA index file to get chromosome lengths."""
    chrom_lengths = {}

    with open(fai_path) as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) >= 2:
                chrom = fields[0]
                length = int(fields[1])
                chrom_lengths[chrom] = length

    return chrom_lengths


def generate_regions(chrom_lengths, region_size, chroms_to_include):
    """Generate regions of specified size for each chromosome.

    Returns list of tuples: (chrom, start, end) where start/end are 1-based.
    """
    regions = []

    # Sort chromosomes: chr1-22 numerically, then chrX, chrY
    def chrom_sort_key(c):
        if c.startswith('chr'):
            c = c[3:]
        if c == 'X':
            return (23, 0)
        elif c == 'Y':
            return (24, 0)
        elif c == 'M' or c == 'MT':
            return (25, 0)
        else:
            try:
                return (int(c), 0)
            except ValueError:
                return (100, c)

    sorted_chroms = sorted(
        [c for c in chrom_lengths.keys() if c in chroms_to_include],
        key=chrom_sort_key
    )

    for chrom in sorted_chroms:
        length = chrom_lengths[chrom]
        start = 1
        while start <= length:
            end = min(start + region_size - 1, length)
            regions.append((chrom, start, end))
            start = end + 1

    return regions


def main():
    args = parse_args()

    # Parse chromosome lengths from input
    if args.vcf:
        chrom_lengths = parse_vcf_contigs(args.vcf)
        if not chrom_lengths:
            print("Error: No ##contig lines found in VCF header.", file=sys.stderr)
            print("Try using --fai with a FASTA index file instead.", file=sys.stderr)
            sys.exit(1)
    else:
        chrom_lengths = parse_fai(args.fai)

    # Determine which chromosomes to include
    if args.chroms:
        chroms_to_include = set(args.chroms.split(','))
    else:
        chroms_to_include = get_default_chroms()

    # Filter to only include requested chromosomes that exist
    available_chroms = set(chrom_lengths.keys())
    chroms_to_include = chroms_to_include & available_chroms

    if not chroms_to_include:
        print("Error: No matching chromosomes found.", file=sys.stderr)
        print(f"Available: {sorted(available_chroms)[:10]}...", file=sys.stderr)
        sys.exit(1)

    # Generate regions
    regions = generate_regions(chrom_lengths, args.region_size, chroms_to_include)

    # Output
    out_file = open(args.output, 'w') if args.output else sys.stdout

    for chrom, start, end in regions:
        if args.format == "bed":
            # BED format: 0-based, tab-separated (chrom, start-1, end)
            print(f"{chrom}\t{start - 1}\t{end}", file=out_file)
        else:
            # Region format: 1-based (chrom:start-end)
            print(f"{chrom}:{start}-{end}", file=out_file)

    if args.output:
        out_file.close()

    # Summary to stderr
    format_desc = "BED (0-based)" if args.format == "bed" else "region (1-based)"
    print(f"Generated {len(regions)} regions of {args.region_size // 1_000_000}MB ({format_desc})", file=sys.stderr)
    print(f"Chromosomes: {len(chroms_to_include)}", file=sys.stderr)


if __name__ == "__main__":
    main()
