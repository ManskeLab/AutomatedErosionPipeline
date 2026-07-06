#!/usr/bin/env python3
"""Match nnUNet erosion crops to the full image they were cut from, by intensity.

The geometry-based matcher (find_match.py) is unreliable, so instead we identify
each crop's source full image by CONTENT: channel _0000 of every erosion crop is
an intensity patch cut from some full image, so it should appear (as a sub-volume)
inside exactly one full image in the reference directory.

For each crop we:
  * orient crop and every full image to a canonical frame (fixes flipped headers),
  * score each full image by normalized cross-correlation (match_template) on a
    downsampled copy for speed,
  * accept the best full image only if its score clears --threshold, else no_match.

Handles the realities the user flagged:
  * a crop may have NO corresponding full image (different modality / not present)
    -> reported as no_match,
  * blank/near-uniform crops -> reported as blank and skipped,
  * mixed modalities (HR-pQCT / CBCT / SR-CBCT) -> the wrong-modality fulls score
    low and are rejected by the threshold.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.feature import match_template
from skimage.measure import block_reduce


def is_nifti(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def oriented_array(img, orient="LPS"):
    """Reorient to a canonical anatomical frame so arrays are directly comparable
    regardless of the stored direction matrix, then return the float32 array."""
    img = sitk.DICOMOrient(img, orient)
    return sitk.GetArrayFromImage(img).astype(np.float32)


def downsample(a, factor):
    """Block-mean downsample (phase-independent, unlike striding) so a true
    sub-volume match survives the coarse search regardless of offset parity."""
    if factor <= 1:
        return a
    return block_reduce(a, (factor, factor, factor), np.mean).astype(np.float32)


def load_crops(imagesTr, channel, factor, blank_std):
    crops = []
    for p in sorted(imagesTr.iterdir()):
        if not is_nifti(p) or f"_{channel}." not in p.name:
            continue
        a = oriented_array(sitk.ReadImage(str(p)))
        ds = downsample(a, factor)
        blank = float(a.std()) < blank_std
        crops.append({
            "path": p, "name": p.name, "ds": ds, "shape": a.shape,
            "blank": blank, "mean": float(a.mean()), "std": float(a.std()),
            "best_ncc": -1.0, "best_full": "", "best_off": None,
        })
    return crops


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imagesTr", required=True, type=Path)
    ap.add_argument("--full-dir", required=True, type=Path,
                    help="Directory of candidate full images (searched recursively)")
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--channel", default="0000", help="Crop channel to match on")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="Min NCC to accept a match (default 0.90)")
    ap.add_argument("--factor", type=int, default=3,
                    help="Downsample factor for the coarse search (default 3)")
    ap.add_argument("--blank-std", type=float, default=1e-6,
                    help="Crops with intensity std below this are 'blank'")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process the first N crops (0 = all)")
    args = ap.parse_args()

    crops = load_crops(args.imagesTr, args.channel, args.factor, args.blank_std)
    if args.limit:
        crops = crops[: args.limit]
    active = [c for c in crops if not c["blank"]]
    print(f"{len(crops)} crops ({len(crops)-len(active)} blank) vs full images in "
          f"{args.full_dir}")

    full_files = sorted(p for p in args.full_dir.rglob("*") if p.is_file() and is_nifti(p))
    print(f"{len(full_files)} candidate full images\n")

    # Loop fulls on the outside so each is loaded/oriented only once.
    for fi, fp in enumerate(full_files, 1):
        try:
            fa = downsample(oriented_array(sitk.ReadImage(str(fp))), args.factor)
        except Exception as e:
            print(f"  [skip full] {fp.name}: {e}")
            continue
        for c in active:
            t = c["ds"]
            if any(ts > fs for ts, fs in zip(t.shape, fa.shape)):
                continue                      # crop bigger than this full -> can't fit
            res = match_template(fa, t)
            peak = float(res.max())
            if peak > c["best_ncc"]:
                off = np.unravel_index(int(np.argmax(res)), res.shape)
                c["best_ncc"] = peak
                c["best_full"] = str(fp.relative_to(args.full_dir))
                c["best_off"] = tuple(int(o * args.factor) for o in off)
        print(f"  [{fi}/{len(full_files)}] scored against {fp.name}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_match = 0
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop", "status", "best_full", "ncc", "approx_offset_zyx",
                    "crop_shape", "mean", "std"])
        for c in crops:
            if c["blank"]:
                status = "blank"
            elif c["best_ncc"] >= args.threshold:
                status = "match"; n_match += 1
            else:
                status = "no_match"
            w.writerow([c["name"], status, c["best_full"],
                        f"{c['best_ncc']:.4f}" if c["best_ncc"] >= 0 else "",
                        c["best_off"] or "", c["shape"],
                        f"{c['mean']:.2f}", f"{c['std']:.2f}"])
    print(f"\nWrote {args.out_csv}: {n_match} match, "
          f"{sum(1 for c in crops if c['blank'])} blank, "
          f"{len(crops)-n_match-sum(1 for c in crops if c['blank'])} no_match")


if __name__ == "__main__":
    main()
