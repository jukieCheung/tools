#!/usr/bin/env python3
"""
Extract line intensity profiles from one or more NIfTI (.nii/.nii.gz) volumes
and save them as a single CSV. Supports arbitrary 3D line between two points
(start -> end), specified either in voxel coordinates (i,j,k) or world/RAS mm.

Examples
--------
# 1) Two PET volumes, line in voxel coords from (30,40,20) to (90,40,20)
python nii_line_profile.py \
  --nii a.nii.gz b.nii.gz \
  --line 30,40,20:90,40,20 \
  --space voxel \
  --step-mm 1.0 \
  --out profile_voxel.csv

# 2) One MRI volume, line in world/RAS mm coords with ~0.5 mm sampling
python nii_line_profile.py \
  --nii mr.nii.gz \
  --line -10.0,-20.0,5.0:40.0,-20.0,5.0 \
  --space world \
  --step-mm 0.5 \
  --out profile_world.csv

# 3) Use time frame 3 for a 4D PET, fallback to nearest-neighbor if SciPy absent
python nii_line_profile.py --nii pet_4d.nii.gz --frame 3 \
  --line 20,30,15:100,30,15 --space voxel --num-samples 200 --out pet_f3.csv

Notes
-----
* If --space voxel is used, the voxel coordinates are interpreted in the
  reference image (the first --nii by default or --ref if provided).
* If sampling points fall outside an image, the CSV will contain NaN there.
* If SciPy is installed, trilinear interpolation is used; otherwise nearest.
"""

import argparse
import csv
import math
import os
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np

try:
    from scipy.ndimage import map_coordinates  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


def parse_triplet(s: str) -> np.ndarray:
    try:
        x, y, z = [float(t) for t in s.split(",")]
        return np.array([x, y, z], dtype=float)
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid triplet '{s}'. Expected format like 'x,y,z' (comma-separated)."
        )


def parse_line(s: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        a, b = s.split(":")
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--line must be 'x,y,z:x2,y2,z2' (colon-separated start:end)."
        )
    p0 = parse_triplet(a)
    p1 = parse_triplet(b)
    return p0, p1


def load_image(path: str, frame: int = 0) -> Tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float64)
    if data.ndim == 4:
        if not (0 <= frame < data.shape[3]):
            raise ValueError(
                f"--frame {frame} is out of range for 4D image '{path}' with {data.shape[3]} frames."
            )
        data = data[..., frame]
    elif data.ndim != 3:
        raise ValueError(f"Unsupported image ndim={data.ndim} for '{path}'.")
    return img, data


def world_from_voxel(vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    homo = np.c_[vox, np.ones((vox.shape[0], 1))]
    ras = (affine @ homo.T).T[:, :3]
    return ras


def voxel_from_world(ras: np.ndarray, affine: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(affine)
    homo = np.c_[ras, np.ones((ras.shape[0], 1))]
    vox = (inv @ homo.T).T[:, :3]
    return vox


def sample_along_line_world(
    img: nib.Nifti1Image,
    data: np.ndarray,
    pts_world: np.ndarray,
    nan_outside: bool = True,
    use_trilinear: bool = True,
) -> np.ndarray:
    vox = voxel_from_world(pts_world, img.affine)
    # bounds check per point
    nx, ny, nz = data.shape
    inside = (
        (vox[:, 0] >= 0)
        & (vox[:, 0] <= nx - 1)
        & (vox[:, 1] >= 0)
        & (vox[:, 1] <= ny - 1)
        & (vox[:, 2] >= 0)
        & (vox[:, 2] <= nz - 1)
    )

    if use_trilinear and _HAS_SCIPY:
        coords = [vox[:, 0], vox[:, 1], vox[:, 2]]
        vals = map_coordinates(
            data,
            coordinates=coords,
            order=1,
            mode="nearest",
            prefilter=False,
        )
    else:
        idx = np.rint(vox).astype(int)
        vals = np.full(len(pts_world), np.nan, dtype=float)
        ok = (
            (idx[:, 0] >= 0)
            & (idx[:, 0] < nx)
            & (idx[:, 1] >= 0)
            & (idx[:, 1] < ny)
            & (idx[:, 2] >= 0)
            & (idx[:, 2] < nz)
        )
        vals[ok] = data[idx[ok, 0], idx[ok, 1], idx[ok, 2]]

    if nan_outside:
        vals[~inside] = np.nan

    return vals


def main():
    ap = argparse.ArgumentParser(
        description="Extract line intensity profiles from NIfTI volumes and save CSV."
    )
    ap.add_argument(
        "--nii",
        nargs="+",
        required=True,
        help="One or more .nii/.nii.gz files.",
    )
    ap.add_argument(
        "--ref",
        default=None,
        help=(
            "Optional reference image path. When --space voxel is used, the voxel"
            " coordinates are interpreted in this image's space. Defaults to the"
            " first --nii file."
        ),
    )
    ap.add_argument(
        "--line",
        type=parse_line,
        required=True,
        metavar="x,y,z:x2,y2,z2",
        help="Start and end points of the line (two triplets, colon-separated).",
    )
    ap.add_argument(
        "--space",
        choices=["voxel", "world"],
        default="voxel",
        help="Coordinate space for --line points: voxel (i,j,k) or world (RAS mm).",
    )
    ap.add_argument(
        "--step-mm",
        type=float,
        default=None,
        help="Sampling step in millimeters along the line (mutually exclusive with --num-samples).",
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples along the line (mutually exclusive with --step-mm).",
    )
    ap.add_argument(
        "--frame",
        type=int,
        default=0,
        help="If an image is 4D, use this frame index (default: 0).",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output CSV path.",
    )
    ap.add_argument(
        "--no-trilinear",
        action="store_true",
        help="Disable trilinear interpolation (use nearest-neighbor).",
    )
    ap.add_argument(
        "--keep-outside",
        action="store_true",
        help="Keep values outside the image bounds (instead of NaN).",
    )

    args = ap.parse_args()

    if (args.step_mm is None) == (args.num_samples is None):
        ap.error("Specify exactly one of --step-mm or --num-samples.")

    # Load images
    imgs: List[nib.Nifti1Image] = []
    datas: List[np.ndarray] = []
    for p in args.nii:
        img, data = load_image(p, frame=args.frame)
        imgs.append(img)
        datas.append(data)

    # Choose reference image for voxel-space interpretation
    if args.ref is not None:
        ref_img, _ = load_image(args.ref, frame=args.frame)
    else:
        ref_img = imgs[0]

    # Prepare line points in world (RAS mm)
    (p0, p1) = args.line
    if args.space == "voxel":
        vox_line = np.vstack([p0, p1])
        ras_line = world_from_voxel(vox_line, ref_img.affine)
        p0w, p1w = ras_line[0], ras_line[1]
    else:
        p0w, p1w = p0, p1

    # Construct sampling points in world space
    vec = p1w - p0w
    length_mm = float(np.linalg.norm(vec))
    if length_mm == 0:
        raise ValueError("Start and end points are identical (zero-length line).")

    if args.num_samples is not None:
        n = int(args.num_samples)
        if n < 2:
            raise ValueError("--num-samples must be >= 2.")
        ts = np.linspace(0.0, 1.0, n)
    else:
        step = float(args.step_mm)
        if step <= 0:
            raise ValueError("--step-mm must be > 0.")
        n = int(math.ceil(length_mm / step)) + 1
        ts = np.linspace(0.0, 1.0, n)

    pts_world = p0w[None, :] + ts[:, None] * vec[None, :]
    s_mm = ts * length_mm

    # Sample each image
    use_trilinear = (not args.no_trilinear) and _HAS_SCIPY
    if (not _HAS_SCIPY) and (not args.no_trilinear):
        print("[INFO] SciPy not found; falling back to nearest-neighbor.")

    profiles = []
    for img, data in zip(imgs, datas):
        vals = sample_along_line_world(
            img, data, pts_world,
            nan_outside=(not args.keep_outside),
            use_trilinear=use_trilinear,
        )
        profiles.append(vals)

    # Write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["s_mm"]
    for p in args.nii:
        name = Path(p).name
        if name.endswith(".nii.gz"):
            name = name[:-7]
        elif name.endswith(".nii"):
            name = name[:-4]
        header.append(name)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(s_mm)):
            row = [f"{s_mm[i]:.6f}"] + [f"{profiles[j][i]:.9g}" for j in range(len(profiles))]
            w.writerow(row)

    # Minimal side info to help reproduce
    meta_txt = out_path.with_suffix(out_path.suffix + ".txt")
    with open(meta_txt, "w") as mf:
        mf.write("Line profile parameters\n")
        mf.write(f"space: {args.space}\n")
        mf.write(f"p0 ({args.space}): {p0.tolist()}\n")
        mf.write(f"p1 ({args.space}): {p1.tolist()}\n")
        mf.write(f"p0_world_mm: {p0w.tolist()}\n")
        mf.write(f"p1_world_mm: {p1w.tolist()}\n")
        mf.write(f"length_mm: {length_mm:.6f}\n")
        if args.num_samples is not None:
            mf.write(f"num_samples: {n}\n")
        else:
            mf.write(f"step_mm: {float(args.step_mm):.6f}\n")
        mf.write(f"frame: {args.frame}\n")
        mf.write(f"interpolation: {'trilinear' if use_trilinear else 'nearest'}\n")
        mf.write("inputs:\n")
        for p in args.nii:
            mf.write(f"  - {p}\n")
        if args.ref is not None:
            mf.write(f"ref: {args.ref}\n")

    print(f"[OK] Wrote CSV to: {out_path}")
    print(f"[OK] Wrote metadata to: {meta_txt}")


if __name__ == "__main__":
    main()
