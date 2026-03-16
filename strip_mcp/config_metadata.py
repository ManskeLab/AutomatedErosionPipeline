#!/usr/bin/env python3
"""
set_spacing.py
Overwrites image metadata (spacing, origin, direction) to match nnUNet plans
without resampling. Can read spacing directly from nnUNetPlans.json.
"""

import argparse
import json
import SimpleITK as sitk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overwrite image spacing metadata to match nnUNet plans (no resampling)."
    )
    parser.add_argument("input", help="Path to input image (will be overwritten)")

    spacing_source = parser.add_mutually_exclusive_group(required=True)
    spacing_source.add_argument(
        "--spacing",
        nargs="+",
        type=float,
        metavar="S",
        help="Spacing in mm, one value per dimension (e.g. --spacing 0.25 0.25 0.25).",
    )
    spacing_source.add_argument(
        "--plans",
        metavar="PATH",
        help="Path to nnUNetPlans.json — spacing will be read from this file.",
    )

    parser.add_argument(
        "--config",
        default="3d_fullres",
        choices=["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"],
        help="Which nnUNet configuration to pull spacing from (default: 3d_fullres).",
    )
    parser.add_argument(
        "--origin",
        nargs="+",
        type=float,
        default=None,
        metavar="O",
        help="Optional: override origin (e.g. --origin 0.0 0.0 0.0).",
    )
    parser.add_argument(
        "--direction",
        nargs="+",
        type=float,
        default=None,
        metavar="D",
        help="Optional: override direction cosines as a flat list (9 values for 3D).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing anything.",
    )
    return parser.parse_args()


def load_spacing_from_plans(plans_path, config):
    with open(plans_path, "r") as f:
        plans = json.load(f)

    config_data = plans["configurations"].get(config)
    if config_data is None:
        raise ValueError(f"Config '{config}' not found in plans.")

    # 3d_cascade_fullres inherits from 3d_fullres
    if "inherits_from" in config_data:
        parent = config_data["inherits_from"]
        print(f"[plans] '{config}' inherits spacing from '{parent}'")
        config_data = plans["configurations"][parent]

    spacing = config_data["spacing"]
    print(f"[plans] Dataset:  {plans['dataset_name']}")
    print(f"[plans] Config:   {config}")
    print(f"[plans] Spacing:  {spacing} mm")
    return spacing


def main():
    args = parse_args()

    if args.plans:
        spacing_list = load_spacing_from_plans(args.plans, args.config)
    else:
        spacing_list = args.spacing

    img = sitk.ReadImage(args.input)
    ndim = img.GetDimension()

    if len(spacing_list) != ndim:
        raise ValueError(
            f"Image is {ndim}D but {len(spacing_list)} spacing values were provided."
        )

    old_spacing   = img.GetSpacing()
    old_origin    = img.GetOrigin()
    old_direction = img.GetDirection()

    new_spacing   = tuple(spacing_list)
    new_origin    = tuple(args.origin)    if args.origin    is not None else old_origin
    new_direction = tuple(args.direction) if args.direction is not None else old_direction

    if args.direction is not None and len(args.direction) != ndim * ndim:
        raise ValueError(
            f"Direction cosines for a {ndim}D image must have {ndim*ndim} values, "
            f"got {len(args.direction)}."
        )

    print(f"\nImage:      {args.input}")
    print(f"Size:       {img.GetSize()}")
    print(f"Spacing:    {old_spacing}  →  {new_spacing}")
    print(f"Origin:     {old_origin}  →  {new_origin}")
    print(f"Direction:  {old_direction}  →  {new_direction}")

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return

    img.SetSpacing(new_spacing)
    img.SetOrigin(new_origin)
    img.SetDirection(new_direction)

    sitk.WriteImage(img, args.input)
    print(f"\nOverwritten: {args.input}")


if __name__ == "__main__":
    main()