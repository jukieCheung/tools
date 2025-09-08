#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将PET图像用3D高斯核平滑到统一的8 mm各向同性FWHM（ADNI风格）

方法学描述（可直接用于稿件）：
我们使用3D高斯核对下载的PET图像进行平滑处理，以调整所有ADNI站点之间相似PET图像的点扩散函数（PSF）。
本研究中使用的平滑核与ADNI数据库中“后处理”图像所使用的相同，后者被命名为“共配准、平均、标准图像以及体素尺寸，统一分辨率”。
平滑后的PET图像具有统一的各向同性分辨率为8毫米全宽半高（FWHM）。

脚本功能：
- 输入：.nii / .nii.gz PET（支持3D或4D动态）
- 输出：平滑至目标FWHM（默认8,8,8 mm）的PET
- 可选已知原始在机FWHM（若提供，将按方差可加计算所需核宽度）
- 不在时间维上平滑（4D时仅对前三个空间维做高斯）

用法示例：
python smooth_to_8mm.py \
  --in PET.nii.gz \
  --out PET_8mm.nii.gz

若已知原始在机分辨率约为6×6×6 mm，目标8 mm：
python smooth_to_8mm.py \
  --in PET_raw.nii.gz \
  --out PET_to8mm.nii.gz \
  --current-fwhm-mm 6 6 6

强制“在已有≥8 mm时仍继续再平滑到8 mm”（一般不建议）：
python smooth_to_8mm.py --in PET.nii.gz --out PET_8mm.nii.gz --force

"""

import argparse
import math
import numpy as np
import nibabel as nib
from nibabel.affines import voxel_sizes
from scipy.ndimage import gaussian_filter


def fwhm_mm_to_sigma_mm(fwhm_mm: float) -> float:
    # FWHM = 2*sqrt(2*ln2) * sigma  =>  sigma = FWHM / 2.354820045...
    return float(fwhm_mm) / 2.3548200450309493


def compute_sigma_kernel_mm(target_fwhm, current_fwhm):
    """
    按方差可加原则：sigma_ker^2 = sigma_tgt^2 - sigma_in^2
    逐轴计算（允许各向异性输入，但目标为各向同性）
    """
    tgt_sig = np.array([fwhm_mm_to_sigma_mm(f) for f in target_fwhm], dtype=float)
    cur_sig = np.array([fwhm_mm_to_sigma_mm(f) for f in current_fwhm], dtype=float)
    sig2 = np.maximum(0.0, tgt_sig**2 - cur_sig**2)
    return np.sqrt(sig2)


def main():
    ap = argparse.ArgumentParser(
        description="Smooth PET NIfTI to 8 mm isotropic FWHM (ADNI-style)."
    )
    ap.add_argument("--in", dest="in_path", required=True, help="Input PET NIfTI (.nii/.nii.gz)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output NIfTI path")
    ap.add_argument(
        "--target-fwhm-mm",
        nargs=3,
        type=float,
        default=[8.0, 8.0, 8.0],
        help="Target FWHM in mm for (x y z). Default: 8 8 8",
    )
    ap.add_argument(
        "--current-fwhm-mm",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        help="Known intrinsic FWHM in mm of input PET (x y z). If unknown, leave as 0 0 0.",
    )
    ap.add_argument(
        "--mode",
        choices=["constant", "nearest", "mirror", "wrap", "reflect"],
        default="constant",
        help="Boundary mode for Gaussian filter. Default: constant (cval=0).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force smoothing even if current FWHM >= target (not recommended).",
    )
    args = ap.parse_args()

    img = nib.load(args.in_path)
    data = img.get_fdata(dtype=np.float32)  # 解码scl_slope/scl_intercept
    hdr = img.header.copy()
    vox = voxel_sizes(img.affine)[:3]  # (dx, dy, dz) in mm, always positive

    # 计算需要的核sigma（mm）
    tgt_fwhm = np.array(args.target_fwhm_mm, dtype=float)
    cur_fwhm = np.array(args.current_fwhm_mm, dtype=float)
    sigKer_mm = compute_sigma_kernel_mm(tgt_fwhm, cur_fwhm)

    # 如果已知current FWHM >= target且未--force，则不平滑（或仅对需要的轴平滑）
    need = sigKer_mm > 0
    if not args.force and not np.any(need):
        print("[INFO] current FWHM >= target in all axes; no smoothing performed.")
        out_img = nib.Nifti1Image(data.astype(np.float32), img.affine, hdr)
        out_img.set_sform(img.get_sform(), code=int(hdr["sform_code"]))
        out_img.set_qform(img.get_qform(), code=int(hdr["qform_code"]))
        out_img.header.set_data_dtype(np.float32)
        nib.save(out_img, args.out_path)
        print(f"[OK] Saved (no-op) to: {args.out_path}")
        return

    # 把sigma从mm换成“体素单位”
    sigKer_vox = np.zeros(3, dtype=float)
    sigKer_vox[need] = sigKer_mm[need] / vox[need]

    # 若是4D，最后一维作为时间维，不做平滑 => sigma=(sx, sy, sz, 0)
    if data.ndim == 4:
        sigma = (sigKer_vox[0], sigKer_vox[1], sigKer_vox[2], 0.0)
    elif data.ndim == 3:
        sigma = (sigKer_vox[0], sigKer_vox[1], sigKer_vox[2])
    else:
        raise ValueError(f"Unsupported data dim {data.ndim}: expected 3D or 4D NIfTI.")

    print("[INFO] Voxel size (mm):", vox)
    print("[INFO] Target FWHM (mm):", tgt_fwhm)
    print("[INFO] Current FWHM (mm):", cur_fwhm)
    print("[INFO] Kernel sigma (mm):", sigKer_mm)
    print("[INFO] Kernel sigma (vox):", sigKer_vox)
    print("[INFO] Applying Gaussian filter, mode =", args.mode)

    smoothed = gaussian_filter(
        data,
        sigma=sigma,
        mode=args.mode,
        cval=0.0,
        truncate=4.0  # ~包含到4σ，足够覆盖8mm核
    ).astype(np.float32)

    out_img = nib.Nifti1Image(smoothed, img.affine, hdr)
    # 保持空间标记
    out_img.set_sform(img.get_sform(), code=int(hdr["sform_code"]))
    out_img.set_qform(img.get_qform(), code=int(hdr["qform_code"]))
    out_img.header.set_data_dtype(np.float32)
    nib.save(out_img, args.out_path)
    print(f"[OK] Saved smoothed image to: {args.out_path}")


if __name__ == "__main__":
    main()
