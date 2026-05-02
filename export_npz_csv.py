"""
export_npz_csv.py
=================
Export isi NTU60_CS_binary_fall.npz ke file CSV.

Setiap baris = 1 sampel skeleton, berisi:
  nama_file, split, performer, kelas_ntu, aksi, label, valid_frames, dll.

Cara pakai:
  python export_npz_csv.py
  python export_npz_csv.py --output hasil.csv
"""

import os
import sys
import argparse
import csv
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
NPZ_PATH  = "data/ntu/NTU60_CS_binary_fall.npz"
STATS_DIR = "data/ntu/statistics"
OUTPUT    = "results/dataset_info.csv"
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_IDS = [1,  2,  4,  5,  8,  9,  13, 14, 15, 16,
             17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
TEST_IDS  = [3,  6,  7,  10, 11, 12, 20, 21, 22, 23,
             24, 26, 29, 30, 32, 33, 36, 37, 39, 40]
SELECTED  = {8, 9, 27, 42, 43}
FALL      = {43}

CLASS_NAMES = {
    8:  "sitting down",
    9:  "standing up",
    27: "jump up",
    42: "staggering",
    43: "falling down",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz",    default=NPZ_PATH)
    p.add_argument("--stats",  default=STATS_DIR)
    p.add_argument("--output", default=OUTPUT)
    return p.parse_args()


def compute_valid_frames(x_ctvm):
    """x_ctvm: (C, T, V, M) → jumlah frame non-nol"""
    return int(np.sum(x_ctvm.sum(0).sum(-1).sum(-1) != 0))


def build_rows_for_split(split, subject_ids, x_raw, y_onehot,
                          names_all, performer_all, label_all):
    """
    Rekonstruksi nama asli + hitung valid_frames tiap sampel.
    Mengembalikan list of dict, satu dict per sampel.
    """
    # Rekonstruksi urutan indeks global — harus sama persis dengan
    # seq_transformation.get_indices() dan prepare_fall_dataset.filter_to_binary()
    indices = []
    for pid in subject_ids:
        indices.extend(np.where(performer_all == pid)[0].tolist())
    indices = np.array(indices, dtype=int)

    labs_1b = label_all[indices]
    mask    = np.isin(labs_1b, list(SELECTED))
    filtered_idx  = indices[mask]
    filtered_labs = labs_1b[mask]
    filtered_names = names_all[filtered_idx]
    filtered_perf  = performer_all[filtered_idx]

    N, T, _ = x_raw.shape
    x_ctvm_all = x_raw.reshape(N, T, 2, 25, 3).transpose(0, 4, 1, 3, 2)

    labels_binary = np.argmax(y_onehot, axis=1)

    if len(filtered_names) != N:
        print(f"  [WARNING] Rekonstruksi nama ({len(filtered_names)}) "
              f"!= jumlah sampel NPZ ({N}) untuk split '{split}' "
              f"→ nama file akan dikosongkan")
        filtered_names = [f"sample_{i:04d}" for i in range(N)]
        filtered_labs  = [int(labels_binary[i]) + 8 for i in range(N)]  # fallback
        filtered_perf  = [0] * N

    rows = []
    for i in range(N):
        ntu_class  = int(filtered_labs[i])
        label_bin  = int(labels_binary[i])
        vf         = compute_valid_frames(x_ctvm_all[i])

        # Parse komponen dari nama file (S001C001P003R001A008)
        fname = str(filtered_names[i])
        setup    = fname[1:4]   if len(fname) >= 20 else ""
        camera   = fname[5:8]   if len(fname) >= 20 else ""
        perf_str = fname[9:12]  if len(fname) >= 20 else ""
        replic   = fname[13:16] if len(fname) >= 20 else ""
        action   = fname[17:20] if len(fname) >= 20 else ""

        rows.append({
            "nama_file"   : fname,
            "split"       : split,
            "setup"       : setup,
            "camera"      : camera,
            "performer_id": perf_str,
            "replication" : replic,
            "action_code" : f"A{action}" if action else "",
            "kelas_ntu"   : ntu_class,
            "aksi"        : CLASS_NAMES.get(ntu_class, ""),
            "label"       : label_bin,
            "label_nama"  : "fall" if label_bin == 1 else "not_fall",
            "valid_frames": vf,
            "T_max"       : T,
            "C"           : 3,
            "V"           : 25,
            "M"           : 2,
        })

    return rows


def main():
    args = parse_args()

    # ── Load stats ────────────────────────────────────────────────────────────
    print(f"Membaca statistics dari: {args.stats}")
    names_all     = np.loadtxt(os.path.join(args.stats, "skes_available_name.txt"), dtype=str)
    performer_all = np.loadtxt(os.path.join(args.stats, "performer.txt"),           dtype=int)
    label_all     = np.loadtxt(os.path.join(args.stats, "label.txt"),              dtype=int)

    # ── Load NPZ ──────────────────────────────────────────────────────────────
    print(f"Membaca NPZ dari: {args.npz}")
    npz = np.load(args.npz)

    # ── Proses tiap split ─────────────────────────────────────────────────────
    all_rows = []

    print("Memproses split TRAIN ...")
    rows_train = build_rows_for_split(
        "train", TRAIN_IDS,
        npz["x_train"], npz["y_train"],
        names_all, performer_all, label_all,
    )
    all_rows.extend(rows_train)
    print(f"  {len(rows_train)} sampel")

    print("Memproses split TEST ...")
    rows_test = build_rows_for_split(
        "test", TEST_IDS,
        npz["x_test"], npz["y_test"],
        names_all, performer_all, label_all,
    )
    all_rows.extend(rows_test)
    print(f"  {len(rows_test)} sampel")

    # ── Tulis CSV ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    fieldnames = list(all_rows[0].keys())
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    total = len(all_rows)
    n_fall    = sum(1 for r in all_rows if r["label"] == 1)
    n_nfall   = total - n_fall

    print(f"\nCSV tersimpan: {args.output}")
    print(f"Total baris   : {total}")
    print(f"  train       : {len(rows_train)}")
    print(f"  test        : {len(rows_test)}")
    print(f"  not_fall    : {n_nfall}")
    print(f"  fall        : {n_fall}")
    print(f"Kolom         : {', '.join(fieldnames)}")


if __name__ == "__main__":
    main()
