#!/usr/bin/env python3
"""Report nonzero voxels + geometry at every pipeline stage for one image, to
localize where the erosion output becomes empty.

Usage (on the cluster):
  python diagnose_pipeline.py \
      --work-dir /work/manske_lab/jobs/actus_erosion/work \
      --input    /work/manske_lab/images/hrpqct/actus/ACTUS_clean/mcp/disease/ACTUS_001/0/ACTUS_001_0_mcp1.nii.gz
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def stat(path):
    if not os.path.exists(path):
        return f"MISSING  {path}"
    try:
        im = sitk.ReadImage(path)
        a = sitk.GetArrayFromImage(im)
        nz = int((a != 0).sum())
        return (f"nz={nz:>9}  size={tuple(im.GetSize())}  "
                f"spacing={tuple(round(x,4) for x in im.GetSpacing())}  "
                f"origin={tuple(round(x,2) for x in im.GetOrigin())}  "
                f"dir={tuple(int(x) for x in im.GetDirection())}  {os.path.basename(path)}")
    except Exception as e:
        return f"READ_ERR {path}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--input", required=True, type=Path)
    args = ap.parse_args()

    key = args.input.name.split(".")[0]                 # ACTUS_001_0_mcp1
    parts = key.split("_")                              # ACTUS 001 <tp...> mcpN
    subject = f"{parts[0]}_{parts[1]}"
    mcp = parts[-1]
    tp = "_".join(parts[2:-1])
    w = args.work_dir / subject / tp / mcp
    print(f"key={key}  work={w}\n")

    print("INPUT     ", stat(str(args.input)))
    print("STRIP     ", stat(str(w / "strip" / f"stripped_{key}.nii.gz")))
    print("  MC_mask ", stat(str(w / "strip" / f"MC_mask_{key}.nii.gz")))
    print("  PP_mask ", stat(str(w / "strip" / f"PP_mask_{key}.nii.gz")))
    print("EDGE      ", stat(str(w / "edge" / f"stripped_{key}.nii.gz")))
    print("CLOSED    ", stat(str(w / "closed" / f"stripped_{key}.nii.gz")))
    print("REG_MC    ", stat(str(w / "reg" / f"ATLAS_TO_stripped_{key}_MC.nii.gz")))
    print("REG_PP    ", stat(str(w / "reg" / f"ATLAS_TO_stripped_{key}_PP.nii.gz")))

    for bone in ("MC", "PP"):
        cand = w / f"cand_{bone}"
        inputs = sorted(glob.glob(str(cand / "*_input*.nii.gz")))
        labeled = glob.glob(str(cand / "*_labeled.nii.gz"))
        print(f"CAND_{bone}   {len(inputs)} ROI(s); labeled:",
              stat(labeled[0]) if labeled else "MISSING")
        for f in inputs:
            print("   ROI    ", stat(f))

    preds = sorted(glob.glob(str(w / "pred" / "*.nii.gz")))
    print(f"PRED      {len(preds)} prediction(s)")
    for f in preds:
        print("   pred   ", stat(f))

    print("\nHint: find the first row above where nz drops to 0 — that stage is "
          "the culprit. If PRED rows are nonzero but the final combined mask is "
          "empty, it's a combine/geometry problem (compare PRED origin/dir to INPUT).")


if __name__ == "__main__":
    main()
