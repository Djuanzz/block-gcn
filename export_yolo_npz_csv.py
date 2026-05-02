"""
export_yolo_npz_csv.py
======================
Export isi skeletons_raw.npz ke CSV.
Satu baris per sampel skeleton.

Cara pakai:
    python export_yolo_npz_csv.py
    python export_yolo_npz_csv.py --npz data/ntu_yolo/skeletons_raw.npz --output hasil.csv
"""

import os
import csv
import argparse
import numpy as np

# ── Default config ─────────────────────────────────────────────────────────
NPZ_PATH = "data/ntu_yolo/skeletons_raw.npz"
OUTPUT   = "results/yolo_dataset_info.csv"

FALL_CLASSES = {43}
CLASS_NAMES  = {
    8:  "sitting down",
    9:  "standing up",
    27: "jump up",
    42: "staggering",
    43: "falling down",
}
CROSS_SUBJECT_TRAIN = {
    1, 2, 4, 5, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
    25, 27, 28, 31, 34, 35, 38
}


def compute_valid_frames(skeleton_T51):
    """skeleton_T51: (T, 51) → jumlah frame non-nol"""
    return int(np.sum(skeleton_T51.sum(axis=1) != 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",    default=NPZ_PATH, help="Path ke skeletons_raw.npz")
    parser.add_argument("--output", default=OUTPUT,   help="Path output CSV")
    args = parser.parse_args()

    print(f"Membaca: {args.npz}")
    data = np.load(args.npz, allow_pickle=True)

    skeletons = data["skeletons"]   # (N, T, 51)
    labels    = data["labels"]      # (N,) action class 1-indexed
    subjects  = data["subjects"]    # (N,) performer ID
    cameras   = data["cameras"]     # (N,) camera ID
    setups    = data["setups"]      # (N,) setup ID
    filenames = data["filenames"]   # (N,) str

    N, T, _ = skeletons.shape
    print(f"Total sampel: {N}  |  T={T}  |  51=(17 joint × 3)")

    rows = []
    for i in range(N):
        fname      = str(filenames[i])
        ntu_class  = int(labels[i])
        performer  = int(subjects[i])
        camera     = int(cameras[i])
        setup      = int(setups[i])

        # Ambil komponen dari nama file (SsssXsssAsss...)
        # Fallback: ambil dari array metadata jika nama tidak bisa di-parse
        stem = fname.replace("_rgb", "").replace(".avi", "").replace(".mp4", "")
        if len(stem) >= 20:
            s_setup  = stem[1:4]
            s_cam    = stem[5:8]
            s_perf   = stem[9:12]
            s_replic = stem[13:16]
            s_action = stem[17:20]
        else:
            s_setup  = f"{setup:03d}"
            s_cam    = f"{camera:03d}"
            s_perf   = f"{performer:03d}"
            s_replic = "001"
            s_action = f"{ntu_class:03d}"

        label_bin  = 1 if ntu_class in FALL_CLASSES else 0
        split      = "train" if performer in CROSS_SUBJECT_TRAIN else "test"
        vf         = compute_valid_frames(skeletons[i])

        rows.append({
            "nama_file"   : fname,
            "split"       : split,
            "setup"       : s_setup,
            "camera"      : s_cam,
            "performer_id": s_perf,
            "replication" : s_replic,
            "action_code" : f"A{s_action}",
            "kelas_ntu"   : ntu_class,
            "aksi"        : CLASS_NAMES.get(ntu_class, ""),
            "label"       : label_bin,
            "label_nama"  : "fall" if label_bin == 1 else "not_fall",
            "valid_frames": vf,
            "T_max"       : T,
            "C"           : 3,
            "V"           : 17,
            "M"           : 1,
        })

    # ── Tulis CSV ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Ringkasan ────────────────────────────────────────────────────────
    n_train  = sum(1 for r in rows if r["split"] == "train")
    n_test   = sum(1 for r in rows if r["split"] == "test")
    n_fall   = sum(1 for r in rows if r["label"] == 1)
    n_nfall  = sum(1 for r in rows if r["label"] == 0)
    vf_vals  = [r["valid_frames"] for r in rows]

    print(f"\nCSV tersimpan : {args.output}")
    print(f"Total baris   : {len(rows)}")
    print(f"  train       : {n_train}")
    print(f"  test        : {n_test}")
    print(f"  not_fall    : {n_nfall}")
    print(f"  fall        : {n_fall}")
    print(f"Valid frames  : min={min(vf_vals)}  max={max(vf_vals)}  "
          f"mean={sum(vf_vals)/len(vf_vals):.1f}")
    print(f"Kolom         : {', '.join(fieldnames)}")


if __name__ == "__main__":
    main()
