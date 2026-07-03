#!/usr/bin/env python3
"""Resample cropped nnUNet erosion ROIs back into their full-size MCP image space.

Each erosion in the nnUNet imagesTr folder is a small crop taken from a full MCP
scan.  This script places every crop back onto the grid of its parent full-size
MCP image (matching size / spacing / origin / direction) so the erosion lines up
with the full bone volume again.

Naming
------
Erosion crop (input) :  mcp<n>_RAIR_<subj>_erosion_<k>[_0000].nii.gz
Full MCP  (reference):  <MCP_ROOT>/RAIR_<subj>_<tp>/mcp<n>_RAIR_<subj>_<tp>.nii.gz

The crop name does NOT contain the timepoint (<tp> = 0 or 1), which is required to
pick the correct reference scan.  The timepoint is recovered from the match CSV
(column `image_file(s)`, e.g. stripped_mcp2_RAIR_001_R_00_erosion1_input.nii.gz).
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import SimpleITK as sitk

INTERPOLATORS = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "bspline": sitk.sitkBSpline,
}

# mcp<n>_<COHORT>_<num>[_<side>][_<tp>]_erosion[_]<k>
NAME_RE = re.compile(
    r"^mcp(?P<mcp>\d+)_(?P<cohort>[A-Za-z]+)_(?P<num>\d+)"
    r"(?:_(?P<side>[RL]))?(?:_(?P<tp>\d{1,2}))?_erosion_?(?P<ero>\d+)$"
)


def clean_stem(filename: str) -> str:
    """Strip .nii.gz, an nnUNet channel suffix (_0000), a stripped_ prefix and
    an _input/_label suffix so we are left with the descriptive core name."""
    stem = filename
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    stem = re.sub(r"_\d{4}$", "", stem)          # nnUNet channel, e.g. _0000
    stem = re.sub(r"^stripped_", "", stem)
    stem = re.sub(r"_(input|label)$", "", stem)
    return stem


def parse_name(stem: str):
    m = NAME_RE.match(stem)
    if not m:
        return None
    d = m.groupdict()
    return {
        "mcp": d["mcp"],
        "cohort": d["cohort"],
        "num": d["num"],
        "tp": None if d["tp"] is None else str(int(d["tp"])),  # 00 -> 0
        "ero": str(int(d["ero"])),
    }


def build_timepoint_lookup(csv_path: Path):
    """(mcp, cohort, num, ero) -> set of timepoints, parsed from the CSV's
    image_file(s) column."""
    lookup = defaultdict(set)
    token_re = re.compile(
        r"mcp(?P<mcp>\d+)_(?P<cohort>[A-Za-z]+)_(?P<num>\d+)_[RL]_"
        r"(?P<tp>\d{1,2})_erosion(?P<ero>\d+)"
    )
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            cell = row.get("image_file(s)") or ""
            for token in cell.split(","):
                m = token_re.search(token.strip())
                if not m:
                    continue
                key = (m["mcp"], m["cohort"], m["num"], str(int(m["ero"])))
                lookup[key].add(str(int(m["tp"])))
    return lookup


def resample_to_reference(src_path, ref_path, out_path, interpolator, default_value):
    src = sitk.ReadImage(str(src_path))
    ref = sitk.ReadImage(str(ref_path))
    out = sitk.Resample(
        src,
        ref,
        sitk.Transform(),          # identity: align by physical coordinates
        interpolator,
        default_value,
        src.GetPixelID(),
    )
    sitk.WriteImage(out, str(out_path), useCompression=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imagesTr", required=True, type=Path,
                    help="nnUNet imagesTr folder containing the erosion crops")
    ap.add_argument("--mcp-root", required=True, type=Path,
                    help="Root holding RAIR_<subj>_<tp>/mcp<n>_... full MCP images")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Where resampled full-size erosions are written")
    ap.add_argument("--csv", required=True, type=Path,
                    help="Match CSV used to recover the timepoint per erosion")
    ap.add_argument("--interp", choices=INTERPOLATORS, default="nearest",
                    help="Interpolator (default: nearest; use linear/bspline for "
                         "grayscale intensity crops)")
    ap.add_argument("--default-value", type=float, default=0.0,
                    help="Fill value outside the crop (default: 0)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would happen without writing anything")
    args = ap.parse_args()

    if not args.imagesTr.is_dir():
        sys.exit(f"ERROR: imagesTr not found: {args.imagesTr}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tp_lookup = build_timepoint_lookup(args.csv)

    files = sorted(p for p in args.imagesTr.iterdir()
                   if p.name.endswith((".nii.gz", ".nii")))
    print(f"Found {len(files)} erosion file(s) in {args.imagesTr}")

    n_ok = n_skip = 0
    for src in files:
        stem = clean_stem(src.name)
        info = parse_name(stem)
        if info is None:
            print(f"  SKIP  {src.name}: name does not parse")
            n_skip += 1
            continue

        tp = info["tp"]
        if tp is None:
            key = (info["mcp"], info["cohort"], info["num"], info["ero"])
            tps = tp_lookup.get(key)
            if not tps:
                print(f"  SKIP  {src.name}: no timepoint in name and none in CSV "
                      f"for {key}")
                n_skip += 1
                continue
            if len(tps) > 1:
                chosen = sorted(tps)[0]
                print(f"  WARN  {src.name}: ambiguous timepoint {sorted(tps)} in "
                      f"CSV; using {chosen}")
                tp = chosen
            else:
                tp = next(iter(tps))

        subj = f"{info['cohort']}_{info['num']}_{tp}"
        ref = args.mcp_root / subj / f"mcp{info['mcp']}_{subj}.nii.gz"
        out = args.out_dir / f"{stem}.nii.gz"

        if not ref.exists():
            print(f"  SKIP  {src.name}: reference missing {ref}")
            n_skip += 1
            continue

        print(f"  OK    {src.name} -> {out.name}  (ref tp={tp})")
        if not args.dry_run:
            resample_to_reference(src, ref, out,
                                  INTERPOLATORS[args.interp], args.default_value)
        n_ok += 1

    print(f"\nDone. {n_ok} resampled, {n_skip} skipped"
          + (" (dry-run, nothing written)" if args.dry_run else ""))
    if n_skip:
        sys.exit(1)


if __name__ == "__main__":
    main()
