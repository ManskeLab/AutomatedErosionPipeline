#!/usr/bin/env python3

import argparse
import os
import nibabel as nib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mask a NIfTI image using a label mask"
    )
    parser.add_argument("--image", required=True, help="Input image (.nii or .nii.gz)")
    parser.add_argument("--mask", required=True, help="Input mask (.nii or .nii.gz)")
    parser.add_argument("--out", required=True, help="Output directory")
    return parser.parse_args()


def strip_nii_suffix(fname):
    if fname.endswith(".nii.gz"):
        return fname[:-7]
    elif fname.endswith(".nii"):
        return fname[:-4]
    else:
        return os.path.splitext(fname)[0]


def main():
    args = parse_args()

    # Load image and mask
    img = nib.load(args.image)
    mask = nib.load(args.mask)

    img_data = img.get_fdata()
    mask_data = mask.get_fdata()

    if img_data.shape != mask_data.shape:
        raise ValueError(
            f"Shape mismatch: image {img_data.shape} vs mask {mask_data.shape}"
        )

    os.makedirs(args.out, exist_ok=True)

    base = strip_nii_suffix(os.path.basename(args.image))

    # ---- full masked image ----
    out_all = os.path.join(args.out, f"stripped_{base}.nii.gz")
    masked_all = img_data * (mask_data > 0)
    nib.save(nib.Nifti1Image(masked_all, img.affine, img.header), out_all)

    # ---- MC (label == 1) ----
    out_mc = os.path.join(args.out, f"MC_stripped_{base}.nii.gz")
    mc_masked = img_data * (mask_data == 1)
    nib.save(nib.Nifti1Image(mc_masked, img.affine, img.header), out_mc)

    # ---- PP (label == 2) ----
    out_pp = os.path.join(args.out, f"PP_stripped_{base}.nii.gz")
    pp_masked = img_data * (mask_data == 2)
    nib.save(nib.Nifti1Image(pp_masked, img.affine, img.header), out_pp)

    # ---- MC (label == 1) ----
    out_mc = os.path.join(args.out, f"MC_mask_{base}.nii.gz")
    mc_masked = mask_data == 1
    nib.save(nib.Nifti1Image(mc_masked, img.affine, img.header), out_mc)

    # ---- PP (label == 2) ----
    out_pp = os.path.join(args.out, f"PP_mask_{base}.nii.gz")
    pp_masked = mask_data == 2
    nib.save(nib.Nifti1Image(pp_masked, img.affine, img.header), out_pp)


if __name__ == "__main__":
    main()