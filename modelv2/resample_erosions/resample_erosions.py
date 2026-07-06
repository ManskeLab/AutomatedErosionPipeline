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

import numpy as np
import SimpleITK as sitk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INTERPOLATORS = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "bspline": sitk.sitkBSpline,
}

# mcp<n>_<COHORT>_<num>[_<side>][_<tp>]_erosion[_]<k>
# Descriptive erosion name (crop already renamed), tp/side optional:
#   mcp2_RAIR_001_erosion_1  |  stripped_mcp2_RAIR_001_R_00_erosion1_input
NAME_RE = re.compile(
    r"^mcp(?P<mcp>\d+)_(?P<cohort>[A-Za-z]+)_(?P<num>\d+)"
    r"(?:_(?P<side>[RL]))?(?:_(?P<tp>\d{1,2}))?_erosion_?(?P<ero>\d+)$"
)

# Descriptive token embedded anywhere (used to read the CSV image_file column):
#   ...mcp2_RAIR_001_R_00_erosion1...   (side + timepoint always present here)
TOKEN_RE = re.compile(
    r"mcp(?P<mcp>\d+)_(?P<cohort>[A-Za-z]+)_(?P<num>\d+)_[RL]_"
    r"(?P<tp>\d{1,2})_erosion(?P<ero>\d+)"
)


def strip_ext(name: str) -> str:
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def clean_stem(filename: str) -> str:
    """Strip .nii.gz, an nnUNet channel suffix (_0000), a stripped_ prefix and
    an _input/_label suffix so we are left with the descriptive core name."""
    stem = strip_ext(filename)
    stem = re.sub(r"_\d{4}$", "", stem)          # nnUNet channel, e.g. _0000
    stem = re.sub(r"^stripped_", "", stem)
    stem = re.sub(r"_(input|label)$", "", stem)
    return stem


def token_info(match):
    """Turn a TOKEN_RE match into a normalized info dict."""
    return {
        "mcp": match["mcp"],
        "cohort": match["cohort"],
        "num": match["num"],
        "tp": str(int(match["tp"])),      # 00 -> 0
        "ero": str(int(match["ero"])),
    }


def parse_name(stem: str):
    """Parse a descriptive crop filename directly (timepoint may be absent)."""
    m = NAME_RE.match(stem)
    if not m:
        return None
    d = m.groupdict()
    return {
        "mcp": d["mcp"],
        "cohort": d["cohort"],
        "num": d["num"],
        "tp": None if d["tp"] is None else str(int(d["tp"])),
        "ero": str(int(d["ero"])),
    }


def build_timepoint_lookup(csv_path: Path):
    """(mcp, cohort, num, ero) -> set of timepoints, from the CSV image_file column.
    Used when the crop filename is descriptive but lacks a timepoint."""
    lookup = defaultdict(set)
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            for token in (row.get("image_file(s)") or "").split(","):
                m = TOKEN_RE.search(token.strip())
                if m:
                    key = (m["mcp"], m["cohort"], m["num"], str(int(m["ero"])))
                    lookup[key].add(str(int(m["tp"])))
    return lookup


def build_label_lookup(csv_path: Path):
    """nnUNet label id (e.g. 'EROSION_071') -> (info, ambiguous?).

    Maps the anonymized nnUNet name to the descriptive source via the CSV's
    label_file column; the image_file token carries mcp/subject/timepoint/erosion.
    """
    lookup = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            label = strip_ext((row.get("label_file") or "").strip())
            if not label:
                continue
            infos = [token_info(m) for token in (row.get("image_file(s)") or "").split(",")
                     for m in [TOKEN_RE.search(token.strip())] if m]
            if not infos:
                continue
            ambiguous = (row.get("status", "").strip() == "ambiguous"
                         or any(i != infos[0] for i in infos[1:]))
            lookup[label] = (infos[0], ambiguous)
    return lookup


def is_empty(img):
    """True if the erosion label has no foreground (all voxels == 0)."""
    stats = sitk.StatisticsImageFilter()
    stats.Execute(img)
    return stats.GetMinimum() == 0 and stats.GetMaximum() == 0


def physical_bounds(img):
    """World-coordinate min/max corner of the whole image grid."""
    size = img.GetSize()
    corners = [img.TransformIndexToPhysicalPoint(
                   (i * (size[0] - 1), j * (size[1] - 1), k * (size[2] - 1)))
               for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    lo = [min(c[d] for c in corners) for d in range(3)]
    hi = [max(c[d] for c in corners) for d in range(3)]
    return lo, hi


def describe_geometry(tag, img):
    lo, hi = physical_bounds(img)
    fmt = lambda v: "[" + ", ".join(f"{x:.3f}" for x in v) + "]"
    print(f"    {tag}")
    print(f"      size     {tuple(img.GetSize())}")
    print(f"      spacing  {fmt(img.GetSpacing())}")
    print(f"      origin   {fmt(img.GetOrigin())}")
    print(f"      direction{fmt(img.GetDirection())}")
    print(f"      world_lo {fmt(lo)}")
    print(f"      world_hi {fmt(hi)}")


def resample_to_reference(src, ref_path, out_path, interpolator, default_value):
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


def resample_to_reference(src, ref_path, out_path, interpolator, default_value):
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
    return out, ref


def save_sagittal_snapshot(ero_img, ref_img, out_png, pad=20):
    """Save a PNG of the sagittal slice with the largest erosion cross-section,
    with the erosion (nonzero voxels) overlaid in red on the MCP bone."""
    ero = sitk.GetArrayFromImage(ero_img)   # (z, y, x)
    ref = sitk.GetArrayFromImage(ref_img).astype(float)
    mask = ero != 0
    if not mask.any():
        return False

    # sagittal plane = perpendicular to x (last array axis); pick the fullest one
    x = int(np.argmax(mask.sum(axis=(0, 1))))
    bg = ref[:, :, x]                        # (z, y)
    fg = mask[:, :, x]

    # crop to a padded bounding box around the erosion so it fills the frame
    zs, ys = np.where(fg)
    z0, z1 = max(zs.min() - pad, 0), min(zs.max() + pad + 1, bg.shape[0])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, bg.shape[1])
    bg, fg = bg[z0:z1, y0:y1], fg[z0:z1, y0:y1]

    # contrast: clip the bone to its 1-99 percentile
    finite = bg[np.isfinite(bg)]
    vmin, vmax = (np.percentile(finite, (1, 99)) if finite.size else (0, 1))

    sx, sy, sz = ero_img.GetSpacing()        # sitk order = (x, y, z)
    aspect = sz / sy if sy else 1.0

    overlay = np.zeros(fg.shape + (4,))
    overlay[fg] = [1, 0, 0, 0.45]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(bg, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, aspect=aspect)
    ax.imshow(overlay, origin="lower", aspect=aspect)
    ax.set_title(f"{out_png.stem}\nsagittal x={x}", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


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
    ap.add_argument("--no-snapshot", action="store_true",
                    help="Do not save the sagittal snapshot PNGs")
    ap.add_argument("--snapshot-dir", type=Path, default=None,
                    help="Where snapshot PNGs go (default: <out-dir>/snapshots)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would happen without writing anything")
    ap.add_argument("--inspect", type=int, default=0, metavar="N",
                    help="Diagnostic: print crop vs reference geometry + physical "
                         "overlap for the first N resolvable erosions, then exit")
    args = ap.parse_args()

    if not args.imagesTr.is_dir():
        sys.exit(f"ERROR: imagesTr not found: {args.imagesTr}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = args.snapshot_dir or (args.out_dir / "snapshots")
    if not args.no_snapshot:
        snap_dir.mkdir(parents=True, exist_ok=True)
    tp_lookup = build_timepoint_lookup(args.csv)
    label_lookup = build_label_lookup(args.csv)

    files = sorted(p for p in args.imagesTr.iterdir()
                   if p.name.endswith((".nii.gz", ".nii")))
    print(f"Found {len(files)} erosion file(s) in {args.imagesTr}")

    n_ok = n_skip = 0
    used_stems = {}          # output stem -> source file, to catch collisions
    for src in files:
        stem = clean_stem(src.name)

        # Resolve to (mcp, cohort, num, tp, ero). Two naming styles are handled:
        #  1) descriptive crop name  -> parse directly; recover tp from CSV if absent
        #  2) anonymized nnUNet name (EROSION_071) -> look up via CSV label column
        info = parse_name(stem)
        if info is not None:
            tp = info["tp"]
            if tp is None:
                key = (info["mcp"], info["cohort"], info["num"], info["ero"])
                tps = tp_lookup.get(key)
                if not tps:
                    print(f"  SKIP  {src.name}: no timepoint in name and none in "
                          f"CSV for {key}")
                    n_skip += 1
                    continue
                tp = sorted(tps)[0]
                if len(tps) > 1:
                    print(f"  WARN  {src.name}: ambiguous timepoint {sorted(tps)}"
                          f" in CSV; using {tp}")
                info = dict(info, tp=tp)
        else:
            found = label_lookup.get(stem)
            if found is None:
                print(f"  SKIP  {src.name}: name does not parse and no CSV match "
                      f"for '{stem}'")
                n_skip += 1
                continue
            info, ambiguous = found
            tp = info["tp"]
            if ambiguous:
                print(f"  WARN  {src.name}: CSV match is ambiguous; using "
                      f"mcp{info['mcp']}_{info['cohort']}_{info['num']}_"
                      f"erosion{info['ero']} (tp {tp})")

        # Skip erosions whose label is empty (no foreground). Only checkable when
        # actually reading the image (i.e. not in --dry-run).
        src_img = None
        if not args.dry_run:
            src_img = sitk.ReadImage(str(src))
            if is_empty(src_img):
                print(f"  SKIP  {src.name}: empty erosion (no foreground)")
                n_skip += 1
                continue

        out_stem = (f"mcp{info['mcp']}_{info['cohort']}_{info['num']}"
                    f"_erosion_{info['ero']}")
        if out_stem in used_stems:                # never silently overwrite
            owner = used_stems[out_stem]
            out_stem = f"{out_stem}_{stem}"
            print(f"  WARN  {src.name}: output name collides with {owner}; "
                  f"writing {out_stem}")
        used_stems[out_stem] = src.name

        subj = f"{info['cohort']}_{info['num']}_{tp}"
        ref = args.mcp_root / subj / f"mcp{info['mcp']}_{subj}.nii.gz"
        out = args.out_dir / f"{out_stem}.nii.gz"

        if not ref.exists():
            print(f"  SKIP  {src.name}: reference missing {ref}")
            n_skip += 1
            continue

        if args.inspect:
            ero_img = src_img if src_img is not None else sitk.ReadImage(str(src))
            ref_img = sitk.ReadImage(str(ref))
            elo, ehi = physical_bounds(ero_img)
            rlo, rhi = physical_bounds(ref_img)
            overlap = all(elo[d] < rhi[d] and ehi[d] > rlo[d] for d in range(3))
            print(f"\n=== {src.name}  ->  {ref.name} ===")
            describe_geometry("EROSION crop:", ero_img)
            describe_geometry("REFERENCE   :", ref_img)
            print(f"    physical boxes overlap: {overlap}")
            n_ok += 1
            if n_ok >= args.inspect:
                break
            continue

        print(f"  OK    {src.name} -> {out.name}  (ref tp={tp})")
        if not args.dry_run:
            out_img, ref_img = resample_to_reference(
                src_img, ref, out, INTERPOLATORS[args.interp], args.default_value)
            if not args.no_snapshot:
                png = snap_dir / f"{out_stem}.png"
                if not save_sagittal_snapshot(out_img, ref_img, png):
                    print(f"  WARN  {src.name}: no foreground after resample, "
                          f"no snapshot")
        n_ok += 1

    print(f"\nDone. {n_ok} resampled, {n_skip} skipped"
          + (" (dry-run, nothing written)" if args.dry_run else ""))
    if n_skip:
        sys.exit(1)


if __name__ == "__main__":
    main()
