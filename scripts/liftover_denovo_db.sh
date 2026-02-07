#!/bin/bash
set -euo pipefail

# Liftover denovo-db SSC DNMs from hg19 to hg38
#
# Usage: bash liftover_denovo_db.sh
#
# Requires: liftOver (UCSC), hg19ToHg38.over.chain.gz

RESOURCES=/expanse/projects/sebat1/s3/data/sebat/nf_synthdnm_ssc/resources
INPUT=$RESOURCES/denovo-db.ssc-samples.variants.v.1.6.1.tsv.gz
CHAIN=$RESOURCES/hg19ToHg38.over.chain.gz
OUTPUT=$RESOURCES/denovo-db.ssc-samples.variants.v.1.6.1.hg38.tsv.gz
TMPDIR=$(mktemp -d)

echo "=== Liftover denovo-db SSC DNMs (hg19 → hg38) ==="

# Step 1: Extract BED from TSV (chr, start=pos-1, end=pos, line_number)
# denovo-db uses bare chromosome numbers (1, 2, ..., X, Y), need "chr" prefix
# Skip 2 header lines (##version and #column header)
echo "Extracting BED coordinates..."
zcat "$INPUT" | awk -F'\t' 'NR>2 {
    chr = "chr" $9
    pos = $10
    print chr "\t" (pos - 1) "\t" pos "\t" NR
}' > "$TMPDIR/hg19.bed"

TOTAL=$(wc -l < "$TMPDIR/hg19.bed")
echo "  $TOTAL variants to lift over"

# Step 2: Run liftOver
echo "Running liftOver..."
liftOver "$TMPDIR/hg19.bed" "$CHAIN" "$TMPDIR/hg38.bed" "$TMPDIR/unmapped.bed" -minMatch=0.95

LIFTED=$(wc -l < "$TMPDIR/hg38.bed")
UNMAPPED=$(grep -c "^[^#]" "$TMPDIR/unmapped.bed" 2>/dev/null || echo 0)
echo "  Lifted: $LIFTED"
echo "  Unmapped: $UNMAPPED"

# Step 3: Build a lookup of line_number -> hg38_chr, hg38_pos
# hg38.bed has: chr, start, end, line_number
# Convert back to 1-based position
echo "Building hg38 coordinate lookup..."
awk -F'\t' '{print $4 "\t" $1 "\t" ($3)}' "$TMPDIR/hg38.bed" | sort -k1,1n > "$TMPDIR/hg38_lookup.tsv"

# Step 4: Join back to original TSV, replacing chr and pos with hg38 coordinates
echo "Joining hg38 coordinates to original data..."
zcat "$INPUT" | awk -F'\t' -v OFS='\t' '
    NR == FNR {
        hg38_chr[int($1)] = $2
        hg38_pos[int($1)] = $3
        next
    }
    FNR <= 2 {
        # Header lines — add genome build note
        if (FNR == 1) print $0
        else print $0 "\tHg38_Chr\tHg38_Position"
        next
    }
    {
        line = FNR
        if (line in hg38_chr) {
            print $0 "\t" hg38_chr[line] "\t" hg38_pos[line]
        }
    }
' "$TMPDIR/hg38_lookup.tsv" - | gzip > "$OUTPUT"

FINAL=$(zcat "$OUTPUT" | wc -l)
echo ""
echo "=== Done ==="
echo "  Output: $OUTPUT"
echo "  Rows (incl headers): $FINAL"
echo "  Dropped (unmapped): $UNMAPPED"

# Cleanup
rm -rf "$TMPDIR"
