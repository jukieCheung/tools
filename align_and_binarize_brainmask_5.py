
import os
import re
import glob
import argparse
from datetime import datetime

import numpy as np
import SimpleITK as sitk


# -------------------- Utilities --------------------

def find_pairs(root):
    pairs = []
    for reg_dir in glob.glob(os.path.join(root, "*", "registered", "*")):
        if not os.path.isdir(reg_dir):
            continue
        t1_candidates = sorted(glob.glob(os.path.join(reg_dir, "*_T1w_128iso2.nii.gz")))
        bm_path = os.path.join(reg_dir, "brainmask_inMR.nii.gz")
        if len(t1_candidates) == 0 or not os.path.exists(bm_path):
            continue
        t1_path = t1_candidates[0]
        subject_dir = os.path.dirname(os.path.dirname(reg_dir))
        session_dirname = os.path.basename(reg_dir)
        pairs.append((t1_path, bm_path, subject_dir, session_dirname))
    return pairs


def find_original_mr(subject_dir, session_dirname, t1_iso_path):
    m = re.search(r'(I\d+)_T1w_128iso2\.nii\.gz$', os.path.basename(t1_iso_path))
    target_id = m.group(1) if m else None

    candidates = []
    for seq_dir in glob.glob(os.path.join(subject_dir, "MP-RAGE*")):
        session_path = os.path.join(seq_dir, session_dirname)
        if not os.path.isdir(session_path):
            continue
        for f in glob.glob(os.path.join(session_path, "*.nii.gz")):
            bn = os.path.basename(f)
            if "_T1w_128iso2" in bn or "aparc" in bn or "mask4d" in bn or "voi" in bn or "mask" in bn:
                continue
            candidates.append(f)

    if not candidates:
        return None

    if target_id:
        for f in candidates:
            if os.path.basename(f).startswith(target_id):
                return f
    return candidates[0]


def otsu_brain_mask(img, close_radius=1):
    m = sitk.OtsuThreshold(img, 0, 1, 200)
    if close_radius and close_radius > 0:
        m = sitk.BinaryMorphologicalClosing(m, [close_radius]*3)
    cc = sitk.ConnectedComponent(m)
    lbl = sitk.RelabelComponent(cc, sortByObjectSize=True)
    brain = sitk.BinaryThreshold(lbl, 1, 1, 1, 0)
    return brain


# -------------------- Registration cores --------------------

def register_rigid_intensity(fixed_img, moving_img, fixed_mask=None, moving_mask=None,
                             iters=800, bins=64, sampling=0.5, init="geometry", log=None):
    if init == "geometry":
        initial = sitk.CenteredTransformInitializer(fixed_img, moving_img, sitk.Euler3DTransform(),
                                                    sitk.CenteredTransformInitializerFilter.GEOMETRY)
    else:
        initial = sitk.CenteredTransformInitializer(fixed_img, moving_img, sitk.Euler3DTransform(),
                                                    sitk.CenteredTransformInitializerFilter.MOMENTS)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)
    if 0.0 < sampling < 1.0:
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(sampling, seed=42)
    else:
        reg.SetMetricSamplingStrategy(reg.NONE)

    if fixed_mask is not None:
        reg.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        reg.SetMetricMovingMask(moving_mask)

    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-5, numberOfIterations=iters, relaxationFactor=0.5)
    reg.SetOptimizerScalesFromPhysicalShift()

    reg.SetInitialTransform(initial, inPlace=False)
    final_tfm = reg.Execute(fixed_img, moving_img)
    if log is not None:
        log.write(f"    metric(final)={reg.GetMetricValue():.6f} stop={reg.GetOptimizerStopConditionDescription()}\n")
    return final_tfm, reg.GetMetricValue()


def register_mask_to_t1(fixed_t1, moving_mask_float, iters=600, bins=32, sampling=0.3, log=None):
    blurred = sitk.SmoothingRecursiveGaussian(moving_mask_float, sigma=1.0)
    return register_rigid_intensity(fixed_t1, blurred, None, None, iters, bins, sampling, "geometry", log)


# -------------------- Binarization --------------------

def binarize_mask(img_float, method="otsu", frac=0.2, eps=0.0, close_radius=2, fill="3d", keep_largest=True):
    """
    method:
      - 'otsu'   : Otsu thresholding
      - 'frac'   : threshold = frac * max
      - 'nonzero': strict mask, 1 where value > eps, else 0
    keep_largest: keep only the largest connected component (ignored if False)
    """
    if method == "nonzero":
        # strictly > eps; default eps ~ nextafter(0,1) if not specified
        thr = eps if eps > 0 else np.nextafter(0.0, 1.0, dtype=np.float32)
        bin_img = sitk.BinaryThreshold(img_float, lowerThreshold=float(thr), upperThreshold=1e12, insideValue=1, outsideValue=0)
    elif method == "otsu":
        bin_img = sitk.OtsuThreshold(img_float, 0, 1, 200)
    else:
        arr = sitk.GetArrayFromImage(img_float)
        mx = float(arr.max())
        thr = frac * mx if mx > 0 else 0.0
        bin_img = sitk.BinaryThreshold(img_float, thr, 1e12, 1, 0)

    if close_radius and close_radius > 0:
        bin_img = sitk.BinaryMorphologicalClosing(bin_img, [close_radius]*3)

    if keep_largest:
        cc = sitk.ConnectedComponent(bin_img)
        lbl = sitk.RelabelComponent(cc, sortByObjectSize=True)
        bin_img = sitk.BinaryThreshold(lbl, 1, 1, 1, 0)

    if fill == "3d":
        bin_img = sitk.BinaryFillhole(bin_img, True)
    elif fill == "2d":
        arr = sitk.GetArrayFromImage(bin_img).astype(np.uint8)
        out = np.zeros_like(arr)
        for z in range(arr.shape[0]):
            sl = sitk.GetImageFromArray(arr[z])
            sl_filled = sitk.BinaryFillhole(sl, True)
            out[z] = sitk.GetArrayFromImage(sl_filled)
        tmp = sitk.GetImageFromArray(out)
        tmp.CopyInformation(bin_img)
        bin_img = tmp

    return sitk.Cast(bin_img, sitk.sitkUInt8)


# -------------------- Pipeline --------------------

def process_subject(t1_iso_path, bm_path, subject_dir, session_dirname,
                    method, frac, eps, close_radius, fill, keep_largest,
                    iters, bins, sampling, init, mode, quick_geom=False, log=None):
    fixed_t1 = sitk.ReadImage(t1_iso_path, sitk.sitkFloat32)
    moving_bm = sitk.ReadImage(bm_path, sitk.sitkFloat32)

    reg_dir = os.path.dirname(t1_iso_path)
    out_affine = os.path.join(reg_dir, "brainmask_inMR_rigid_to_T1.tfm")
    out_float  = os.path.join(reg_dir, "brainmask_inMR_rigid_to_T1.nii.gz")
    out_bin    = os.path.join(reg_dir, "brainmask_inMR_rigid_to_T1_bin.nii.gz")

    if log:
        log.write(f"\nProcessing session: {subject_dir} | {session_dirname}\n")
        log.write(f"  fixed T1iso: {t1_iso_path}\n  moving mask: {bm_path}\n")

    used_mode = mode
    tfm = None

    if quick_geom:
        moving_ref = None
        mr_orig = find_original_mr(subject_dir, session_dirname, t1_iso_path) if mode in ("auto","mr2t1") else None
        if mr_orig is not None:
            if log: log.write(f"  [quick-geom] Using orig MR for geometric initializer: {mr_orig}\n")
            moving_ref = sitk.ReadImage(mr_orig, sitk.sitkFloat32)
            used_mode = "quick-geom(mr)"
        else:
            if log: log.write("  [quick-geom] Using mask for geometric initializer.\n")
            moving_ref = moving_bm
            used_mode = "quick-geom(mask)"

        tfm = sitk.CenteredTransformInitializer(
            fixed_t1, moving_ref, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

    if tfm is None:
        if mode in ("auto", "mr2t1"):
            mr_orig = find_original_mr(subject_dir, session_dirname, t1_iso_path)
            if mr_orig is not None:
                if log:
                    log.write(f"  Found orig MR: {mr_orig}\n  -> register MR->T1iso then apply to mask\n")
                moving_mr = sitk.ReadImage(mr_orig, sitk.sitkFloat32)
                fixed_mask = otsu_brain_mask(fixed_t1, close_radius=1)
                moving_mask = otsu_brain_mask(moving_mr, close_radius=1)

                tfm_geo, m_geo = register_rigid_intensity(fixed_t1, moving_mr, fixed_mask, moving_mask,
                                                          iters=iters, bins=bins, sampling=sampling, init=init, log=log)
                tfm_mom, m_mom = register_rigid_intensity(fixed_t1, moving_mr, fixed_mask, moving_mask,
                                                          iters=iters, bins=bins, sampling=sampling,
                                                          init=("moments" if init=="geometry" else "geometry"), log=log)
                tfm = tfm_mom if m_mom < m_geo else tfm_geo
                used_mode = "mr2t1"
            elif mode == "mr2t1":
                if log:
                    log.write("  [WARN] --mode mr2t1 but original MR not found; skipping.\n")
                return

        if tfm is None:
            if log: log.write("  Fallback to mask2t1 registration.\n")
            tfm, _ = register_mask_to_t1(fixed_t1, moving_bm, iters=max(400, iters//2), bins=max(32, bins//2),
                                         sampling=min(0.5, sampling), log=log)
            used_mode = "mask2t1"

    resampled = sitk.Resample(
        moving_bm, fixed_t1, tfm, sitk.sitkNearestNeighbor, 0.0, sitk.sitkFloat32
    )
    sitk.WriteImage(resampled, out_float, True)
    sitk.WriteTransform(tfm, out_affine)

    bin_mask = binarize_mask(resampled, method=method, frac=frac, eps=eps,
                             close_radius=close_radius, fill=fill, keep_largest=keep_largest)
    bin_mask.CopyInformation(fixed_t1)
    sitk.WriteImage(bin_mask, out_bin, True)

    if log:
        nz = int(sitk.GetArrayViewFromImage(bin_mask).sum())
        voxvol = np.prod(bin_mask.GetSpacing())
        log.write(f"  Mode used: {used_mode}\n")
        log.write(f"  Saved transform: {out_affine}\n  Aligned (float): {out_float}\n  Binary: {out_bin}\n")
        log.write(f"  Mask voxels: {nz}  (~{nz*voxvol:.2f} mm^3)\n")


# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="Rigidly align brainmask_inMR to *_T1w_128iso2; strict nonzero binarization available.")
    ap.add_argument("--root", required=True, help="Root folder (e.g., /m/g/1/ADNI_OUT)")
    ap.add_argument("--mode", choices=["auto", "mr2t1", "mask2t1"], default="auto",
                    help="auto: prefer MR->T1 then apply to mask; mr2t1: require MR; mask2t1: old behavior")
    ap.add_argument("--quick-geom", action="store_true", help="Skip iterative registration; use center-based geometry initializer only")

    ap.add_argument("--method", choices=["otsu", "frac", "nonzero"], default="otsu", help="Binarization method")
    ap.add_argument("--frac", type=float, default=0.2, help="If method=frac, threshold = frac * max")
    ap.add_argument("--eps", type=float, default=0.0, help="If method=nonzero, treat values > eps as 1")
    ap.add_argument("--close-radius", type=int, default=2, help="Binary morphological closing radius (voxels)")
    ap.add_argument("--fill", choices=["none","2d","3d"], default="3d", help="Fill interior holes (2D slice-wise or 3D)")
    ap.add_argument("--keep-largest", dest="keep_largest", action="store_true", default=True, help="Keep only largest connected component")
    ap.add_argument("--no-keep-largest", dest="keep_largest", action="store_false", help="Keep all components (do not drop small ones)")

    ap.add_argument("--iters", type=int, default=800, help="Registration max iterations")
    ap.add_argument("--bins", type=int, default=64, help="Mattes MI histogram bins")
    ap.add_argument("--sampling", type=float, default=0.5, help="Metric sampling percentage (0< s <1, or 1 for dense)")
    ap.add_argument("--init", choices=["geometry","moments"], default="geometry", help="Transform initializer for full registration")
    ap.add_argument("--log", type=str, default=None, help="Path to save a run log")
    args = ap.parse_args()

    log = open(args.log, "a", encoding="utf-8") if args.log else None
    if log:
        log.write(f"\n=== Run at {datetime.now().isoformat(timespec='seconds')} ===\n")

    pairs = find_pairs(args.root)
    if log:
        log.write(f"Found {len(pairs)} sessions under {args.root}\n")

    for t1_iso_path, bm_path, subj_dir, sess_dir in pairs:
        try:
            process_subject(t1_iso_path, bm_path, subj_dir, sess_dir,
                            method=args.method, frac=args.frac, eps=args.eps,
                            close_radius=args.close_radius, fill=args.fill, keep_largest=args.keep_largest,
                            iters=args.iters, bins=args.bins, sampling=args.sampling, init=args.init,
                            mode=args.mode, quick_geom=args.quick_geom, log=log)
        except Exception as e:
            if log:
                log.write(f"  ERROR: {e}\n")
            else:
                print(f"[ERROR] {t1_iso_path}: {e}")

    if log:
        log.write("Done.\n")
        log.close()


if __name__ == "__main__":
    main()
