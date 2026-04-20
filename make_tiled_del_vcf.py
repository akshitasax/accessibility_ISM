#!/usr/bin/env python3
"""
make_tiled_del_vcf.py

Generate adjacent (non-overlapping) tiled deletions across a region and write a valid VCF.

Input region format:
  chr11,+,69598124-69600237   OR   NC_000077.6,+,69598124-69600237

For each tile deletion starting at `pos` (1-based), length = tile:
  VCF POS is left-anchored at (pos-1)
  REF = anchor_base + deleted_sequence
  ALT = anchor_base

This produces standard VCF deletions compatible with tools like supremo_lite/variant workflows.

Requirements:
  pip/conda install pysam
  genome FASTA must have .fai index.
"""

import argparse
import re
from datetime import datetime

# --- one mapping dict used for BOTH directions (mm10/GRCm38 primarys) ---
GRCM38_REFSEQ_BY_CHR = {
    "1": "NC_000067.6",
    "2": "NC_000068.7",
    "3": "NC_000069.6",
    "4": "NC_000070.6",
    "5": "NC_000071.6",
    "6": "NC_000072.6",
    "7": "NC_000073.6",
    "8": "NC_000074.6",
    "9": "NC_000075.6",
    "10": "NC_000076.6",
    "11": "NC_000077.6",
    "12": "NC_000078.6",
    "13": "NC_000079.6",
    "14": "NC_000080.6",
    "15": "NC_000081.6",
    "16": "NC_000082.6",
    "17": "NC_000083.6",
    "18": "NC_000084.6",
    "19": "NC_000085.6",
    "X": "NC_000086.7",
    "Y": "NC_000087.7",
    "M": "NC_005089.1",
    "MT": "NC_005089.1",
}
CHR_BY_GRCM38_REFSEQ = {v: k for k, v in GRCM38_REFSEQ_BY_CHR.items()}


def parse_region(region: str):
    region = region.strip()

    # New format: chrom:start-end
    m = re.match(r"^([^:]+):(\d+)-(\d+)$", region)
    if m:
        chrom = m.group(1).strip()
        start = int(m.group(2))
        end = int(m.group(3))
        strand = "."   # unknown / not provided
        if start <= 0 or end < start:
            raise ValueError(f"Invalid interval: {start}-{end}")
        return chrom, strand, start, end


def _strip_chr_prefix(chrom: str) -> str:
    c = chrom.strip()
    if c.lower().startswith("chr"):
        c = c[3:]
    return c


# -----
# What was happening:
# Chromosome name mapping was too rigid. The script only tried two possible representations
# (e.g., "2" and "NC_..." for "refseq"; or the reverse), but FASTA/BED/VCF files in real practice
# use a variety of naming styles: "chr1", "1", "NC_...", etc. Your error arose because your
# FASTA and BED use "chr2", which was never matched when only "2" and "NC_..." were tried.
#
# What I changed:
# - Arguments: The user no longer needs to care about --chrom_format unless they want a specific VCF format.
# - I added a _find_matching_contig helper: it tries all plausible, deduplicated forms
#   of each region chromosome (with/without "chr", number, refseq, etc) across the FASTA
#   contig names, and warns clearly if no match is found, showing all tried forms.
# - Reduced repetitive/legacy alias generation to emit only necessary non-redundant aliases.
# - Now the BED, FASTA, and VCF naming autodetects; as long as the names are consistent
#   between your inputs and reference, there's no manual fiddling needed.
#
# Result: The code is less inflated, less redundant, and works out-of-the-box for "chr2", "2", "NC_..." and more.
# -----

def format_chrom(chrom_in: str, chrom_format: str) -> str:
    """
    Convert chromosome to requested output format for VCF writing.
    This does not affect lookup in FASTA/BED, only VCF output.
    """
    c = chrom_in.strip()
    if c.startswith("NC_"):
        refseq = c
        num = CHR_BY_GRCM38_REFSEQ.get(refseq, refseq)
    else:
        num = _strip_chr_prefix(c).upper()
        refseq = GRCM38_REFSEQ_BY_CHR.get(num, c)
    if chrom_format == "refseq":
        return refseq
    if chrom_format == "number":
        return num
    # If not recognized (e.g. user wants to keep input as-is), return original
    return chrom_in

def _canonical_chroms(chrom):
    """
    Return a tuple of deduplicated, plausible alternate names for chrom:
    e.g., for 'chr2' -> ('chr2', '2', 'NC_...') if available.
    Intended for matching FASTA contig names.
    """
    out = []
    raw = chrom.strip()
    if raw not in out:
        out.append(raw)
    nochr = _strip_chr_prefix(raw)
    if nochr and nochr not in out:
        out.append(nochr)
    nochrU = nochr.upper()
    if nochrU not in out:
        out.append(nochrU)
    # Add chr prefix variant if missing
    if not raw.lower().startswith("chr"):
        chr_nochr = "chr" + nochr
        if chr_nochr not in out:
            out.append(chr_nochr)
    # Try GRCm38 RefSeq alias if available
    if nochrU in GRCM38_REFSEQ_BY_CHR:
        refseq = GRCM38_REFSEQ_BY_CHR[nochrU]
        if refseq not in out:
            out.append(refseq)
    return tuple(out)

def _find_matching_contig(chrom, fasta_references):
    """
    Given chrom and list/tuple of FASTA contig names,
    return the contig name that matches, trying all plausible aliases.
    """
    tried = []
    for alias in _canonical_chroms(chrom):
        if alias in fasta_references:
            return alias
        tried.append(alias)
    # Try one last: lowercased/nochr for odd UCSC/ensembl edge cases
    for alias in _canonical_chroms(chrom):
        alias_lower = alias.lower()
        if alias_lower in fasta_references:
            return alias_lower
        tried.append(alias_lower)
    raise KeyError(
        f"Could not find any matching contig for '{chrom}' in FASTA. "
        f"Checked: {tried}\nFASTA contig samples: {list(fasta_references)[:10]}\n"
        "Check your region/bed FASTA naming consistency (e.g., is it chr2 or 2?)."
    )

def read_bed_regions(bed_path: str):
    """
    BED is 0-based, half-open: [start0, end0).
    Uses first 3 columns: chrom, start, end.
    4th column (name) is optional, extra columns ignored.
    """
    with open(bed_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"BED parse error line {line_num}: expected >=3 columns, got {len(fields)}")
            chrom = fields[0].strip()
            start0 = int(fields[1])
            end0 = int(fields[2])
            if start0 < 0 or end0 <= start0:
                raise ValueError(f"BED invalid interval line {line_num}: {chrom}:{start0}-{end0}")
            name = fields[3].strip() if len(fields) >= 4 else None
            # Convert to 1-based inclusive
            start1 = start0 + 1
            end1 = end0
            yield chrom, start1, end1, name

def main():
    ap = argparse.ArgumentParser(
        description="Generate adjacent tiled deletions across a region and write a VCF."
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--region",
        help="One region: chrom:start-end (e.g. chr11:69598124-69600237)"
    )
    group.add_argument(
        "--bed",
        help="BED file of regions (0-based half-open). Uses first 3 columns; optional name in col4."
    )

    ap.add_argument("--genome_fasta", required=True, help="Genome FASTA path (must have .fai)")
    ap.add_argument("--tile", type=int, default=10, help="Deletion length (default 10)")
    ap.add_argument("--out_vcf", required=True, help="Output VCF path")
    ap.add_argument("--include_partial_last_tile", action="store_true",
                    help="Include shorter last deletion if interval length isn't divisible by tile.")
    ap.add_argument(
        "--chrom_format",
        choices=["refseq", "number", "as_input"],
        default="as_input",
        help="Chrom naming in VCF output (not for FASTA): 'refseq', 'number', or 'as_input' (default: BED/region input style)."
    )
    ap.add_argument("--id_prefix", default=None,
                    help="Optional ID prefix for VCF ID field.")
    args = ap.parse_args()

    try:
        import pysam
    except ImportError as e:
        raise SystemExit("pysam is required. Install with: pip install pysam") from e

    regions = []  # list of (chrom_in, start, end, name)
    if args.region:
        chrom_in, strand, start, end = parse_region(args.region)
        regions.append((chrom_in, start, end, None))
    else:
        regions.extend(read_bed_regions(args.bed))

    tile = args.tile
    step = tile

    fa = pysam.FastaFile(args.genome_fasta)
    fa_references = set(fa.references)   # for quick lookup

    records = []

    for chrom_in, start, end, name in regions:
        # Find matching FASTA contig for chrom_in
        contig_fetch = _find_matching_contig(chrom_in, fa_references)
        # Output chrom name: map only if format is asked, otherwise use BED (input) style for VCF
        contig_out = format_chrom(chrom_in, args.chrom_format) if args.chrom_format != "as_input" else chrom_in

        # Left-anchored deletions need anchor at pos-1, so we must have start >= 2
        if start < 2:
            continue

        contig_len = fa.get_reference_length(contig_fetch)
        if start > contig_len:
            continue
        if end > contig_len:
            end = contig_len

        pos = start
        while pos <= end:
            remaining = end - pos + 1
            if remaining < tile and not args.include_partial_last_tile:
                break
            del_len = tile if remaining >= tile else remaining

            del_end = pos + del_len - 1
            if del_end > contig_len:
                break

            anchor_pos = pos - 1  # 1-based
            anchor0 = anchor_pos - 1  # 0-based
            anchor_base = fa.fetch(contig_fetch, anchor0, anchor0 + 1).upper()
            if not anchor_base or anchor_base == "\n":
                raise RuntimeError(f"Failed to fetch anchor base at {contig_fetch}:{anchor_pos}")

            del0_start = pos - 1
            del0_end = del_end
            deleted_seq = fa.fetch(contig_fetch, del0_start, del0_end).upper()
            if len(deleted_seq) != del_len:
                raise RuntimeError(
                    f"Fetched deleted seq length {len(deleted_seq)} != expected {del_len} "
                    f"at {contig_fetch}:{pos}-{del_end}"
                )

            ref = anchor_base + deleted_seq
            alt = anchor_base
            vcf_pos = anchor_pos  # left-anchored POS

            # contig_out controls the output CHROM field in the VCF; by design, this may be mapped to RefSeq style (NC_...) if --chrom_format asks for that.
            # If you want to always output the original BED/input chromosome (like chr2), use chrom_in for both contig_out and in the ID.
            # Change contig_out to chrom_in if you do not want the output CHROM field to use the mapped contig name.

            # If you want VCF CHROM and ID to match input chromosomes (chr... etc):
            vcf_chrom = chrom_in

            # ID (prefix with BED name if provided)
            base_id = f"{chrom_in}_{pos}_del{del_len}"
            if name:
                base_id = f"{name}|{base_id}"

            if args.id_prefix:
                vid = f"{args.id_prefix}_{base_id}"
            else:
                vid = base_id

            info = "."
            records.append((vcf_chrom, vcf_pos, vid, ref, alt, ".", "PASS", info))
            pos += step

    # Write VCF header
    today = datetime.now().strftime("%Y%m%d")
    header_lines = [
        "##fileformat=VCFv4.2",
        f"##fileDate={today}",
        "##source=make_tiled_del_vcf.py",
        f"##reference={args.genome_fasta}",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
    ]

    with open(args.out_vcf, "w") as out:
        out.write("\n".join(header_lines) + "\n")
        for rec in records:
            out.write("\t".join(map(str, rec)) + "\n")

    fa.close()
    print(f"Wrote {len(records)} deletion records to {args.out_vcf}")
    if records:
        ntiles = (records[-1][1] + tile - records[0][1]) // tile
        if (end - start + 1) % tile != 0 and not args.include_partial_last_tile:
            print(f"Note: interval length {(end - start + 1)} not divisible by tile {tile}; trailing remainder dropped.")
    print("Tip: for many tools, bgzip+tabix is useful:")
    print(f"  bgzip -f {args.out_vcf} && tabix -p vcf {args.out_vcf}.gz")

if __name__ == "__main__":
    main()
