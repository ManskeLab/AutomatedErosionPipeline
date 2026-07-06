#!/usr/bin/env python3
"""Combine per-erosion nnUNet predictions back into full MCP-image space.

Each prediction is a small mask that (thanks to the fixed crop origin in
segm_erosion.py) carries its true position in the original image. We resample
every prediction onto the reference MCP grid and merge them into a single mask
in the same space/size as the input MCP image.
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", required=True, type=Path,
                    help="Folder of per-erosion prediction masks for one image")
    ap.add_argument("--ref", required=True, type=Path,
                    help="Reference defining the resample grid (the 1mm stripped "
                         "image the predictions live in)")
    ap.add_argument("--geom-ref", type=Path, default=None,
                    help="Optional: stamp this image's spacing/origin/direction "
                         "onto the output (the original native-spacing MCP). Must "
                         "be the same voxel grid as --ref.")
    ap.add_argument("--out", required=True, type=Path, help="Output mask path")
    ap.add_argument("--binary", action="store_true",
                    help="Write a binary mask instead of per-erosion instance labels")
    args = ap.parse_args()

    ref = sitk.ReadImage(str(args.ref))
    geom = sitk.ReadImage(str(args.geom_ref)) if args.geom_ref else ref
    out = np.zeros(sitk.GetArrayFromImage(ref).shape, dtype=np.uint16)  # (z,y,x)

    preds = sorted(p for p in args.pred_dir.glob("*.nii.gz")) if args.pred_dir.is_dir() else []
    n_placed = 0
    for inst, p in enumerate(preds, start=1):
        pred = sitk.ReadImage(str(p))
        res = sitk.Resample(pred, ref, sitk.Transform(),
                            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        a = sitk.GetArrayFromImage(res) > 0
        if not a.any():
            print(f"  {p.name}: empty after resample, skipped")
            continue
        label = 1 if args.binary else inst
        fill = a & (out == 0)                 # don't overwrite an earlier erosion
        out[fill] = label
        n_placed += 1
        print(f"  {p.name}: placed {int(a.sum())} voxels as label {label}")

    if geom.GetSize() != ref.GetSize():
        raise SystemExit(f"--geom-ref size {geom.GetSize()} != --ref size "
                         f"{ref.GetSize()}; must be the same voxel grid")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    res_img = sitk.GetImageFromArray(out)
    res_img.CopyInformation(geom)          # true (native) spacing/origin/direction
    sitk.WriteImage(res_img, str(args.out), useCompression=True)
    print(f"Wrote {args.out}: {n_placed} erosion(s) from {len(preds)} prediction(s)")


if __name__ == "__main__":
    main()
