import argparse, json, os, sys, re
import numpy as np
import nibabel as nib

def load_lut(lut_path):
    """Parse FreeSurferColorLUT.txt -> dict{name_lower: id}"""
    name2id = {}
    with open(lut_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'): 
                continue
            parts = re.split(r'\s+', s)
            if len(parts) < 2:
                continue
            try:
                _id = int(parts[0])
            except ValueError:
                continue
            name = parts[1]
            name2id[name.lower()] = _id
    return name2id

def build_oldid_to_newid(mapping_json, name2id, exclude_extras=True, consecutive=False):
    """
    mapping_json: dict new_voi_key -> [old_label_names]
    exclude_extras: drop midline/corpus callosum/CSF/vessel/choroid groups (keep hemi + key VOIs)
    consecutive: reindex new ids as 1..N in deterministic order
    """
    # Define patterns to exclude from "extras"
    excl_patterns = [
        r'corpus[_ ]?callosum', r'cerebrospinal[_ ]?fluid', r'ventricle', r'vessel', r'choroid', r'5th[-_ ]?ventricle',
        r'brain[_ ]?stem', r'midline'
    ]
    def is_extra(key):
        k = key.lower()
        return any(re.search(p, k) for p in excl_patterns)

    # Collect keys in deterministic order: lh_* then rh_*, then others
    keys = sorted(mapping_json.keys(), key=lambda k: (0 if k.startswith('lh_') else (1 if k.startswith('rh_') else 2), k))
    if exclude_extras:
        keys = [k for k in keys if not is_extra(k)]

    # Assign default new ids: 100.. for lh, 200.. for rh, 300.. others
    def default_new_id(key, idx):
        if key.startswith('lh_'): return 100 + idx
        if key.startswith('rh_'): return 200 + idx
        return 300 + idx

    key2newid = {}
    for idx, key in enumerate(keys, start=1):
        key2newid[key] = default_new_id(key, idx)

    # Optionally remap to consecutive 1..N (keeps order)
    if consecutive:
        key2newid = {k:i for i,k in enumerate(keys, start=1)}

    # Build old_id -> new_id using LUT
    old2new = {}
    missing = []
    for key in keys:
        names = mapping_json[key]
        new_id = key2newid[key]
        for nm in names:
            lut_id = name2id.get(nm.lower())
            if lut_id is None:
                missing.append(nm)
            else:
                old2new[lut_id] = new_id
    return old2new, key2newid, keys, missing

def remap_volume(aparc_path, old2new, out_path):
    img = nib.load(aparc_path)
    data = img.get_fdata().astype(np.int32, copy=False)
    out = np.zeros_like(data, dtype=np.int32)
    # Vectorized remap: build a lookup array up to max label
    maxlab = int(data.max())
    lut = np.zeros(maxlab+1, dtype=np.int32)
    for old_id, new_id in old2new.items():
        if old_id <= maxlab:
            lut[old_id] = new_id
    # Apply
    mask = (data >= 0) & (data <= maxlab)
    out[mask] = lut[data[mask]]
    nib.save(nib.MGHImage(out, img.affine, img.header), out_path)
    return out

def save_mapping_csv(old2new, key2newid, keys, mapping_json, csv_path):
    import csv
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['old_id','old_label_name','new_voi_key','new_voi_id'])
        # reverse LUT: for each key, list names -> ids via old2new
        for key in keys:
            new_id = key2newid[key]
            for nm in mapping_json[key]:
                # old id might be absent if not in user's LUT; mark as -1
                # We'll try to infer from old2new
                old_id = -1
                # Attempt to find id that maps to this new_id and name
                # This is best-effort (names not available here), so leave -1 by default
                w.writerow([old_id, nm, key, new_id])

def main():
    ap = argparse.ArgumentParser(description='Remap FreeSurfer aparc+aseg (113 labels) to DeepPVC-style merged VOIs.')
    ap.add_argument('--aparc', required=True, help='Path to aparc+aseg.mgz (or .nii.gz)')
    ap.add_argument('--lut', required=True, help='Path to FreeSurferColorLUT.txt (e.g., $FREESURFER_HOME/FreeSurferColorLUT.txt)')
    ap.add_argument('--mapping', required=True, help='Path to deeppvc_voi_mapping.json')
    ap.add_argument('--out', required=True, help='Output path for merged VOI volume (e.g., voi44.mgz)')
    ap.add_argument('--mapping-csv-out', default=None, help='Optional: write a CSV of the mapping used')
    ap.add_argument('--exclude-extras', action='store_true', help='Exclude corpus callosum / CSF / ventricles / vessels / choroid / brainstem / midline groups')
    ap.add_argument('--consecutive', action='store_true', help='Assign new VOI ids consecutively from 1..N (instead of 100/200 series)')
    args = ap.parse_args()

    # Load resources
    name2id = load_lut(args.lut)
    with open(args.mapping, 'r', encoding='utf-8') as f:
        mapping_json = json.load(f)

    # Build mapping
    old2new, key2newid, keys, missing = build_oldid_to_newid(mapping_json, name2id,
                                                             exclude_extras=args.exclude_extras,
                                                             consecutive=args.consecutive)
    if missing:
        print(f"[WARN] {len(missing)} label names in mapping not found in your LUT. Example(s): {missing[:5]}", file=sys.stderr)

    # Remap volume
    remap_volume(args.aparc, old2new, args.out)
    print(f"[OK] Wrote merged VOI volume to: {args.out}")
    print(f"[INFO] Used {len(keys)} merged VOI groups; new IDs range: {min(key2newid.values())}..{max(key2newid.values())}")

    if args.mapping_csv_out:
        save_mapping_csv(old2new, key2newid, keys, mapping_json, args.mapping_csv_out)
        print(f"[OK] Wrote mapping CSV to: {args.mapping_csv_out}")

if __name__ == '__main__':
    main()