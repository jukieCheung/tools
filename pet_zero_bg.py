#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero PET background (post-processing) using MR-space masks, without re-running PoR.

Reads a PoR summary CSV (from batch_por_all.py) and for each PET output:
  - Finds a mask in the *same MR folder* (defaults tried in order):
      mask4d_from_aparc44.nii.gz
      mask4d_44voi.nii.gz
      mask4d_from_aparc.nii.gz
      brainmask_inMR.nii.gz
  - If the mask is 4D, unions across frames (>0) to a 3D mask.
  - Sets PET voxels outside the mask to 0.
  - Optionally zeroes tiny residual values (|x| < --eps) and negatives.
  - Saves RESULT next to the PET as <stem>_bg0.nii.gz (or in-place if --inplace).

Usage
-----
python pet_zero_bg.py \
  --csv "/mnt/g/11c_pib_mri/outputs/PoRsummary.csv" \
  --eps 1e-6 \
  --which both \
  --inplace   # optional; otherwise writes *_bg0.nii.gz

Options
-------
--csv          Summary CSV path from batch_por_all.py
--which        pet|rbv|both  (fix PET_in_MR final, RBV PVC result, or both) [default: both]
--mask-name    Override default mask filename (searched in MR folder)
--eps          Values with |x|<eps are set to 0 [default: 0]
--dry-run      Don't write files; print what would be done
"""

import argparse, csv, os
from pathlib import Path
import numpy as np
import nibabel as nib

DEFAULT_MASK_CANDIDATES = [
    "mask4d_from_aparc44.nii.gz",
    "mask4d_44voi.nii.gz",
    "mask4d_from_aparc.nii.gz",
    "brainmask_inMR.nii.gz"
]

def read_csv_rows(csv_path: Path):
    import pandas as pd
    # robust read
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(csv_path)
    # normalize columns
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    rows = df.to_dict("records")
    return rows

def union_mask(mask_path: Path) -> nib.Nifti1Image:
    m = nib.load(str(mask_path))
    data = m.get_fdata()
    if data.ndim == 4:
        uni = (data > 0).any(axis=3).astype(np.uint8)
    else:
        uni = (data > 0).astype(np.uint8)
    out = nib.Nifti1Image(uni, m.affine, m.header)
    return out

def apply_mask(pet_path: Path, mask_img: nib.Nifti1Image, eps: float, inplace: bool) -> Path:
    pet = nib.load(str(pet_path))
    pet_data = pet.get_fdata().astype(np.float32)
    m = mask_img.get_fdata().astype(bool)
    if pet_data.ndim == 4:
        # Broadcast mask
        m4 = np.repeat(m[..., None], pet_data.shape[3], axis=3)
        pet_data[~m4] = 0.0
    else:
        pet_data[~m] = 0.0
    if eps and eps > 0:
        pet_data[np.abs(pet_data) < eps] = 0.0
    # guard against tiny negatives
    pet_data[pet_data < 0] = 0.0

    out_img = nib.Nifti1Image(pet_data, pet.affine, pet.header)
    out_img.header.set_xyzt_units("mm", "sec")
    if inplace:
        out_path = pet_path
    else:
        if "".join(pet_path.suffixes).lower().endswith(".nii.gz"):
            stem = pet_path.name[:-7]
            out_name = f"{stem}_bg0.nii.gz"
        else:
            out_name = f"{pet_path.stem}_bg0.nii.gz"
        out_path = pet_path.parent / out_name
    nib.save(out_img, str(out_path))
    return out_path

def find_mask(mr_path: Path, override_name: str | None) -> Path | None:
    folder = mr_path.parent
    candidates = [override_name] if override_name else DEFAULT_MASK_CANDIDATES
    for name in candidates:
        if not name:
            continue
        p = folder / name
        if p.exists():
            return p
    return None

def main():
    ap = argparse.ArgumentParser(description="Zero PET background using MR-space mask (no re-run)")
    ap.add_argument("--csv", required=True, type=Path, help="PoR summary CSV")
    ap.add_argument("--which", default="both", choices=["pet","rbv","both"], help="Which outputs to fix [both]")
    ap.add_argument("--mask-name", default=None, help="Mask filename to look for in MR folder (overrides defaults)")
    ap.add_argument("--eps", type=float, default=0.0, help="Zero values with |x| < eps [0]")
    ap.add_argument("--inplace", action="store_true", help="Overwrite PET file instead of writing *_bg0.nii.gz")
    ap.add_argument("--dry-run", action="store_true", help="Print actions only")
    args = ap.parse_args()

    rows = read_csv_rows(args.csv)

    # column name guesses
    col_mr = None
    for c in ["mr","mr_path","t1_img","t1w","nifti_resampled"]:
        if c in rows[0]:
            col_mr = c; break
    col_petfinal = None
    for c in ["pet_in_mr_final","pet_in_mr_PoR_final","pet_in_mr_final_path"]:
        if c in rows[0]:
            col_petfinal = c; break
    col_rbv = None
    for c in ["rbv_pvc_path","pet_pvc_RBV_in_mr","rbv_in_mr_path"]:
        if c in rows[0]:
            col_rbv = c; break

    if col_mr is None:
        raise SystemExit("[ERROR] Cannot find MR column in CSV (tried mr, mr_path, t1_img, t1w, nifti_resampled)")
    if col_petfinal is None and args.which in ("pet","both"):
        print("[WARN] No PET-final column found; skipping PET final")
    if col_rbv is None and args.which in ("rbv","both"):
        print("[WARN] No RBV column found; skipping RBV")

    for r in rows:
        mr_path = Path(str(r.get(col_mr,"")).strip())
        if not mr_path.exists():
            print(f"[SKIP] MR missing: {mr_path}")
            continue
        mask_path = find_mask(mr_path, args.mask_name)
        if mask_path is None:
            print(f"[SKIP] Mask not found in {mr_path.parent}")
            continue

        mask_img = union_mask(mask_path)

        targets = []
        if args.which in ("pet","both") and col_petfinal and r.get(col_petfinal):
            targets.append(Path(str(r[col_petfinal]).strip()))
        if args.which in ("rbv","both") and col_rbv and r.get(col_rbv):
            targets.append(Path(str(r[col_rbv]).strip()))

        for pet_file in targets:
            if not pet_file.exists():
                print(f"[SKIP] PET missing: {pet_file}")
                continue
            if args.dry_run:
                print(f"[DRY] would mask: {pet_file.name}  using  {mask_path.name}  eps={args.eps} inplace={args.inplace}")
            else:
                out_path = apply_mask(pet_file, mask_img, args.eps, args.inplace)
                print(f"[OK]  masked: {pet_file.name} -> {out_path.name}")

if __name__ == "__main__":
    main()
