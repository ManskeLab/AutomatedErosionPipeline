#!/usr/bin/env python3
"""Run the full atlas-guided erosion segmentation pipeline on a dataset and
submit every stage as SLURM jobs with the correct dependencies.

Input layout (unstripped MCP joint images), e.g.:
    <input>/ACTUS_001/0/ACTUS_001_0_mcp1.nii.gz
            ACTUS_001/2/ACTUS_001_2_mcp3.nii.gz
            ACTUS_001/2_precision/ACTUS_001_2_precision_mcp2.nii.gz

Per image the job graph is:

    strip ─┬─ edge ── closed_edge ─┐
           ├─ reg(MC) ─────────────┼─ candidate(MC) ─┐
           └─ reg(PP) ─────────────┴─ candidate(PP) ─┴─ predict ── combine

Both bones (MC + PP) are processed and merged. Output mirrors the input layout:
    <out>/ACTUS_001/0/ACTUS_001_0_mcp1_erosions.nii.gz
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

# stage launcher scripts
S_STRIP   = REPO / "strip_mcp"        / "predict_edge.sh"
S_EDGE    = REPO / "edge_masking"     / "predict_edge.sh"
S_CLOSED  = REPO / "edge_masking"     / "predict_closed_edge.sh"
S_REG     = REPO / "registration"     / "reg_atlas_to_img.sh"
S_CAND    = REPO / "candidate_erosions" / "run_candidate.sh"
S_PREDICT = REPO / "erosion_model"    / "predict_batch.sh"
S_COMBINE = REPO / "combine"          / "combine_erosions.sh"


class Submitter:
    def __init__(self, dry_run):
        self.dry_run = dry_run
        self._fake = 0

    def submit(self, script, args, deps=None, name=None):
        cmd = ["sbatch", "--parsable"]
        if deps:
            cmd.append("--dependency=afterok:" + ":".join(str(d) for d in deps))
        if name:
            cmd += ["--job-name", name]
        cmd.append(str(script))
        cmd += [str(a) for a in args]
        if self.dry_run:
            self._fake += 1
            print(f"[dry-run] jid={self._fake}  {' '.join(cmd)}")
            return self._fake
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            sys.exit(f"sbatch failed:\n{' '.join(cmd)}\n{out.stderr}")
        jid = out.stdout.strip().split(";")[0]
        print(f"  submitted jid={jid}  {script.name} {' '.join(str(a) for a in args)}")
        return jid


def discover(input_dir, subjects, timepoints, mcps):
    """Yield (subject, tp, mcp_n, image_path)."""
    for subj_dir in sorted(input_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        if subjects and subj_dir.name not in subjects:
            continue
        for tp in timepoints:
            tp_dir = subj_dir / tp
            if not tp_dir.is_dir():
                continue
            for img in sorted(tp_dir.glob("*mcp*.nii*")):
                m = re.search(r"mcp(\d+)", img.name)
                if not m:
                    continue
                n = int(m.group(1))
                if mcps and n not in mcps:
                    continue
                yield subj_dir.name, tp, n, img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path,
                    help="Where all intermediate stage outputs go")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Final combined erosion masks (mirrors input layout)")
    ap.add_argument("--atlas-root", type=Path,
                    default=Path("/work/manske_lab/images/hrpqct/mcp_atlasses"),
                    help="Holds mcp<N>/atlas_mc_aligned.nii.gz and atlas_pp.nii.gz")
    ap.add_argument("--timepoints", default="0,2,2_precision",
                    help="Comma list of timepoint subfolders to process")
    ap.add_argument("--subjects", default="",
                    help="Comma list of subject folders to restrict to (default all)")
    ap.add_argument("--mcps", default="1,2,3,4,5",
                    help="Comma list of MCP joint numbers to process")
    ap.add_argument("--sr", action="store_true",
                    help="SR-CBCT/CBCT mode (passes --sr to candidate erosions). "
                         "Omit for HR-pQCT.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print sbatch commands without submitting")
    args = ap.parse_args()

    timepoints = [t for t in args.timepoints.split(",") if t]
    subjects = {s for s in args.subjects.split(",") if s}
    mcps = {int(x) for x in args.mcps.split(",") if x}
    sr_flag = "sr" if args.sr else "nosr"

    sub = Submitter(args.dry_run)
    images = list(discover(args.input_dir, subjects, timepoints, mcps))
    if not images:
        sys.exit(f"No images found under {args.input_dir}")
    print(f"Found {len(images)} MCP image(s) to process\n")

    n_ok = 0
    for subject, tp, n, img in images:
        key = img.name.split(".")[0]                 # ACTUS_001_0_mcp1
        atlas_mc = args.atlas_root / f"mcp{n}" / "atlas_mc_aligned.nii.gz"
        atlas_pp = args.atlas_root / f"mcp{n}" / "atlas_pp.nii.gz"

        w = args.work_dir / subject / tp / f"mcp{n}"
        d_strip, d_edge, d_closed = w / "strip", w / "edge", w / "closed"
        d_reg = w / "reg"
        d_cand_mc, d_cand_pp = w / "cand_MC", w / "cand_PP"
        d_ero_in, d_pred = w / "ero_in", w / "pred"
        for d in (d_strip, d_edge, d_closed, d_reg, d_cand_mc, d_cand_pp):
            if not args.dry_run:
                d.mkdir(parents=True, exist_ok=True)

        if not args.dry_run and not (atlas_mc.exists() and atlas_pp.exists()):
            print(f"  SKIP {key}: atlas missing ({atlas_mc} / {atlas_pp})")
            continue

        stripped = d_strip / f"stripped_{key}.nii.gz"
        mc_mask  = d_strip / f"MC_mask_{key}.nii.gz"
        pp_mask  = d_strip / f"PP_mask_{key}.nii.gz"
        edge_mask   = d_edge   / f"stripped_{key}.nii.gz"
        closed_mask = d_closed / f"stripped_{key}.nii.gz"
        reg_mc = d_reg / f"ATLAS_TO_stripped_{key}_MC.nii.gz"
        reg_pp = d_reg / f"ATLAS_TO_stripped_{key}_PP.nii.gz"
        out_file = args.out_dir / subject / tp / f"{key}_erosions.nii.gz"

        print(f"# {key}")
        j_strip  = sub.submit(S_STRIP,  [img, d_strip], name=f"strip_{key}")
        j_edge   = sub.submit(S_EDGE,   [stripped, d_edge], deps=[j_strip],
                              name=f"edge_{key}")
        j_closed = sub.submit(S_CLOSED, [stripped, edge_mask, d_closed],
                              deps=[j_edge], name=f"closed_{key}")
        j_reg_mc = sub.submit(S_REG, [stripped, mc_mask, atlas_mc, d_reg, "MC"],
                              deps=[j_strip], name=f"regMC_{key}")
        j_reg_pp = sub.submit(S_REG, [stripped, pp_mask, atlas_pp, d_reg, "PP"],
                              deps=[j_strip], name=f"regPP_{key}")
        j_cand_mc = sub.submit(S_CAND,
                               [stripped, reg_mc, edge_mask, closed_mask,
                                d_cand_mc, sr_flag],
                               deps=[j_closed, j_reg_mc], name=f"candMC_{key}")
        j_cand_pp = sub.submit(S_CAND,
                               [stripped, reg_pp, edge_mask, closed_mask,
                                d_cand_pp, sr_flag],
                               deps=[j_closed, j_reg_pp], name=f"candPP_{key}")
        j_pred = sub.submit(S_PREDICT,
                            [d_cand_mc, d_cand_pp, d_ero_in, d_pred, key],
                            deps=[j_cand_mc, j_cand_pp], name=f"pred_{key}")
        sub.submit(S_COMBINE, [d_pred, img, out_file],
                   deps=[j_pred], name=f"comb_{key}")
        n_ok += 1
        print()

    print(f"Queued pipelines for {n_ok} image(s)"
          + (" (dry-run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
