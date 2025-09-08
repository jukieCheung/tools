# Create a ready-to-run script that clamps negatives to 0 and caps values at the 99th percentile.
# The script supports 3D and 4D NIfTI. It preserves affine/qform/sform and saves float32 output.
#script_path = "/mnt/data/clip_neg_and_cap_p99.py"
#script = r'''#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
clip_neg_and_cap_p99.py
Set negative voxels to 0, then cap values at the given upper percentile (default 99%).
Works for 3D and 4D NIfTI. Saves a new file (float32) and preserves spatial metadata.
Intended for visualization; keep the original for quantitative analysis.
"""

import argparse, os, sys
import numpy as np
import nibabel as nib

def outname(p, suffix):
    base = os.path.basename(p)
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]
    return os.path.join(os.path.dirname(p), f"{base}{suffix}.nii.gz")

def cap_volume(vol, p=99.0):
    """Set negatives to 0, compute p-th percentile, then clip to [0, pctl]."""
    vol = vol.astype(np.float32, copy=False)
    vol = np.where(np.isfinite(vol), vol, 0.0)
    vol[vol < 0] = 0.0
    # Percentile computed after zeroing negatives (robust to small negative tails)
    flat = vol.reshape(-1)
    pctl = float(np.percentile(flat, p))
    if pctl <= 0:
        pctl = 1e-6
    np.clip(vol, 0.0, pctl, out=vol)
    return vol, pctl

def process_file(inp, p=99.0, per_volume=True, suffix=None, out=None):
    img = nib.load(inp)
    data = np.asanyarray(img.dataobj)  # lazy
    if suffix is None:
        suffix = f"_neg0_capP{int(p)}"
    if out is None:
        out = outname(inp, suffix)

    if data.ndim == 3:
        out_data, pctl = cap_volume(data.astype(np.float32), p)
        p_str = f"{pctl:.6g}"
    elif data.ndim == 4:
        vols = []
        p_list = []
        if per_volume:
            for t in range(data.shape[3]):
                v, pctl = cap_volume(data[..., t].astype(np.float32), p)
                vols.append(v)
                p_list.append(pctl)
        else:
            # global percentile across all timepoints
            arr = data.astype(np.float32)
            arr = np.where(np.isfinite(arr), arr, 0.0)
            arr[arr < 0] = 0.0
            pctl = float(np.percentile(arr, p))
            for t in range(arr.shape[3]):
                v = arr[..., t]
                np.clip(v, 0.0, pctl, out=v)
                vols.append(v)
            p_list = [pctl] * data.shape[3]
        out_data = np.stack(vols, axis=3)
        p_str = f"mean={np.mean(p_list):.6g}, min={np.min(p_list):.6g}, max={np.max(p_list):.6g}"
    else:
        raise ValueError(f"Unsupported dimensionality: {data.ndim}")

    hdr = img.header.copy()
    hdr.set_data_dtype(np.float32)
    # Preserve sform/qform
    s_aff, s_code = img.get_sform(coded=True)
    q_aff, q_code = img.get_qform(coded=True)
    hdr.set_sform(s_aff, code=s_code)
    hdr.set_qform(q_aff, code=q_code)

    nib.save(nib.Nifti1Image(out_data.astype(np.float32, copy=False), img.affine, hdr), out)
    return out, p_str

def main():
    ap = argparse.ArgumentParser("Clamp negatives to 0 and cap to the 99th percentile (visualization copy).")
    ap.add_argument("inp", nargs="+", help="Input NIfTI (.nii or .nii.gz)")
    ap.add_argument("--p", type=float, default=99.9, help="Upper percentile (default 99)")
    ap.add_argument("--global", dest="global_mode", action="store_true",
                    help="For 4D inputs, use a single global percentile across all timepoints (default is per-volume).")
    ap.add_argument("--out", default=None, help="Explicit output path (only if single input).")
    args = ap.parse_args()

    if args.out and len(args.inp) != 1:
        print("--out 只能在单个输入文件时使用", file=sys.stderr)
        sys.exit(2)

    for inp in args.inp:
        out, pinfo = process_file(inp, p=args.p, per_volume=not args.global_mode, out=args.out)
        print(f"Saved -> {out} | percentile info: {pinfo}")

if __name__ == "__main__":
    sys.exit(main())

