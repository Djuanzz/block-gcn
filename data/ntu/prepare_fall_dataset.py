"""
prepare_fall_dataset.py
=======================
Script utama untuk menyiapkan dataset fall detection biner dari NTU-RGB+D 60.

PIPELINE LENGKAP (jalankan berurutan dari folder data/ntu/):
------------------------------------------------------------
  Jika belum punya NTU60_CS.npz (perlu file .skeleton mentah):
    1. python get_raw_skes_data.py       → raw_data/raw_skes_data.pkl
    2. python get_raw_denoised_data.py   → denoised_data/raw_denoised_joints.pkl
    3. python seq_transformation.py      → NTU60_CS.npz  (60 kelas)

  Kemudian jalankan script ini (hanya butuh NTU60_CS.npz):
    4. python prepare_fall_dataset.py    → NTU60_CS_binary_fall.npz  (2 kelas)

KELAS YANG DIPILIH:
------------------------------------------------------------
  Label 0 — non-fall:
    A008  sitting down
    A009  standing up (from sitting position)
    A027  jump up
    A042  staggering
  Label 1 — fall:
    A043  falling down

FORMAT OUTPUT NPZ:
------------------------------------------------------------
  x_train : (N_train, T=300, 150)
            150 = 2 orang × 25 joint × 3 koordinat (x, y, z)
  y_train : (N_train, 2)   — one-hot: [non-fall, fall]
  x_test  : (N_test,  T=300, 150)
  y_test  : (N_test,  2)   — one-hot: [non-fall, fall]

MENJALANKAN TRAINING SETELAH INI:
------------------------------------------------------------
  cd BlockGCN/
  python main.py --config ./config/nturgbd-cross-subject/default.yaml
"""

import numpy as np
import os
import sys

# ─── Konfigurasi ────────────────────────────────────────────────────────────────

INPUT_NPZ  = "./NTU60_CS.npz"
OUTPUT_NPZ = "./NTU60_CS_binary_fall.npz"

# Nomor kelas NTU60 (1-based) yang digunakan
NONFALL_CLASSES = [8, 9, 27, 42]   # kelas tidak jatuh
FALL_CLASSES    = [43]             # kelas jatuh
SELECTED_CLASSES = NONFALL_CLASSES + FALL_CLASSES

CLASS_NAMES = {
    8:  "sitting down",
    9:  "standing up (from sitting)",
    27: "jump up",
    42: "staggering",
    43: "falling down",
}

# ─── Utilitas ───────────────────────────────────────────────────────────────────

def sep(char="═", n=62):
    print(char * n)


def print_split_stats(x, y_onehot, split_name, src_labels):
    """Cetak statistik distribusi kelas setelah filtering."""
    binary = np.argmax(y_onehot, axis=1)
    total  = len(binary)

    sep("─")
    print(f"  [{split_name}]  {total} sampel")
    sep("─")
    print(f"  {'Kelas':>6}  {'Nama Action':38}  {'Jumlah':>7}  {'%':>6}  {'Label'}")
    sep("─")
    for c in SELECTED_CLASSES:
        c0   = c - 1
        cnt  = int((src_labels == c0).sum())
        pct  = 100.0 * cnt / total
        lbl  = 1 if c in FALL_CLASSES else 0
        tag  = "fall" if lbl == 1 else "non-fall"
        print(f"  A{c:03d}   {CLASS_NAMES[c]:38}  {cnt:>7}  {pct:>5.1f}%  {lbl} ({tag})")
    sep("─")
    n_fall    = int((binary == 1).sum())
    n_nonfal  = int((binary == 0).sum())
    ratio     = n_fall / n_nonfal if n_nonfal > 0 else float('inf')
    print(f"  {'Jumlah  non-fall  (label 0)':46}  {n_nonfal:>7}  {100.*n_nonfal/total:>5.1f}%")
    print(f"  {'Jumlah  fall      (label 1)':46}  {n_fall:>7}  {100.*n_fall/total:>5.1f}%")
    print(f"  Rasio fall/non-fall  : 1 : {n_nonfal/n_fall:.1f}" if n_fall > 0
          else "  Rasio  : N/A (tidak ada sampel fall)")
    print(f"  Shape skeleton       : {x.shape}")


def filter_to_binary(x, y_onehot, split_name):
    """
    Filter skeleton ke 5 kelas yang dipilih dan buat label biner.

    Parameter
    ---------
    x        : (N, T, 150)  — data skeleton
    y_onehot : (N, 60)      — one-hot label 60 kelas

    Return
    ------
    x_filt   : (N_filt, T, 150)
    y_binary : (N_filt, 2)   — one-hot [non-fall, fall]
    src_lbl  : (N_filt,)     — indeks kelas asli (0-based)
    """
    # one-hot → integer 0-based
    labels_int  = np.argmax(y_onehot, axis=1)

    selected_0  = [c - 1 for c in SELECTED_CLASSES]
    fall_0      = [c - 1 for c in FALL_CLASSES]

    # filter baris yang classnya termasuk
    mask        = np.isin(labels_int, selected_0)
    x_filt      = x[mask]
    src_lbl     = labels_int[mask]

    # label biner: 1 = fall, 0 = non-fall
    binary_lbl  = np.array([1 if l in fall_0 else 0 for l in src_lbl], dtype=np.int32)

    # one-hot 2-kelas
    y_binary    = np.eye(2, dtype=y_onehot.dtype)[binary_lbl]

    print_split_stats(x_filt, y_binary, split_name, src_lbl)
    return x_filt, y_binary, src_lbl


def verify_output(path):
    """
    Verifikasi bahwa file NPZ yang tersimpan memiliki format yang benar.
    Mengembalikan True jika semua pemeriksaan lolos.
    """
    sep("═")
    print("  VERIFIKASI OUTPUT")
    sep("─")

    if not os.path.exists(path):
        print(f"  [GAGAL] File tidak ditemukan: {path}")
        return False

    npz = np.load(path)
    required_keys = ['x_train', 'y_train', 'x_test', 'y_test']
    ok = True

    # 1. Cek semua key ada
    for key in required_keys:
        if key not in npz.files:
            print(f"  [GAGAL] Key '{key}' tidak ada dalam file NPZ")
            ok = False

    if not ok:
        return False

    # 2. Cek shape dan tipe
    for key in required_keys:
        arr = npz[key]
        print(f"  {key:12}  shape={str(arr.shape):30}  dtype={arr.dtype}")

    # 3. Cek bahwa y adalah one-hot dengan 2 kelas
    for key in ['y_train', 'y_test']:
        y = npz[key]
        if y.shape[1] != 2:
            print(f"  [GAGAL] {key}: harusnya 2 kelas, dapat {y.shape[1]}")
            ok = False
        elif not (y.sum(axis=1) == 1).all():
            print(f"  [GAGAL] {key}: bukan one-hot yang valid")
            ok = False

    # 4. Cek tidak ada NaN / Inf
    for key in ['x_train', 'x_test']:
        x = npz[key]
        nan_c = int(np.isnan(x).sum())
        inf_c = int(np.isinf(x).sum())
        if nan_c > 0 or inf_c > 0:
            print(f"  [PERINGATAN] {key}: {nan_c} NaN, {inf_c} Inf ditemukan")
            ok = False

    # 5. Cek x dan y jumlah sampelnya sama
    if npz['x_train'].shape[0] != npz['y_train'].shape[0]:
        print("  [GAGAL] x_train dan y_train jumlah baris berbeda")
        ok = False
    if npz['x_test'].shape[0] != npz['y_test'].shape[0]:
        print("  [GAGAL] x_test dan y_test jumlah baris berbeda")
        ok = False

    sep("─")
    if ok:
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  Ukuran file    : {size_mb:.1f} MB")
        print(f"  ✓  Semua pemeriksaan LOLOS")
    else:
        print("  ✗  Ada pemeriksaan yang GAGAL — periksa log di atas")

    sep("═")
    return ok


# ─── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    sep("═")
    print("  BlockGCN — Persiapan Dataset Binary Fall Detection")
    print("  NTU-RGB+D 60 | Cross-Subject Split | 2 Kelas")
    sep("═")

    # ── 1. Cek file input ──────────────────────────────────────────────────
    print(f"\n  Input  : {os.path.abspath(INPUT_NPZ)}")
    print(f"  Output : {os.path.abspath(OUTPUT_NPZ)}")

    if not os.path.exists(INPUT_NPZ):
        sep("═")
        print(f"\n  [ERROR] File tidak ditemukan: {INPUT_NPZ}")
        print("""
  Untuk membuat NTU60_CS.npz dari awal, jalankan berurutan:

    1. python get_raw_skes_data.py
       → Membutuhkan folder nturgb+d_skeletons/ (unduh dari situs resmi NTU)
       → Output: raw_data/raw_skes_data.pkl

    2. python get_raw_denoised_data.py
       → Input : raw_data/raw_skes_data.pkl
       → Output: denoised_data/raw_denoised_joints.pkl

    3. python seq_transformation.py
       → Input : denoised_data/raw_denoised_joints.pkl + statistics/
       → Output: NTU60_CS.npz  (60 kelas, shape N×300×150)
  """)
        sys.exit(1)

    # ── 2. Muat NTU60_CS.npz ──────────────────────────────────────────────
    print("\n  Membaca NTU60_CS.npz ...")
    npz     = np.load(INPUT_NPZ)
    x_train = npz['x_train']
    y_train = npz['y_train']
    x_test  = npz['x_test']
    y_test  = npz['y_test']
    print(f"  Shape asli: x_train={x_train.shape}, y_train={y_train.shape}")
    print(f"  Shape asli: x_test ={x_test.shape},  y_test ={y_test.shape}")

    # Validasi format NPZ asli
    assert x_train.ndim == 3, \
        f"x_train harus 3D (N,T,150), dapat shape {x_train.shape}"
    assert x_train.shape[2] == 150, \
        f"Dimensi terakhir harus 150 (2×25×3), dapat {x_train.shape[2]}"
    assert y_train.ndim == 2 and y_train.shape[1] == 60, \
        f"y_train harus (N,60), dapat {y_train.shape}"

    # ── 3. Filter & buat label biner ──────────────────────────────────────
    sep("═")
    print("  FILTERING + LABELING BINER")
    print()
    x_tr_f, y_tr_f, src_tr = filter_to_binary(x_train, y_train, "TRAIN")
    print()
    x_te_f, y_te_f, src_te = filter_to_binary(x_test,  y_test,  "TEST")

    # ── 4. Simpan ─────────────────────────────────────────────────────────
    print()
    sep("═")
    print(f"  Menyimpan ke: {OUTPUT_NPZ}")
    np.savez_compressed(
        OUTPUT_NPZ,
        x_train=x_tr_f,
        y_train=y_tr_f,
        x_test=x_te_f,
        y_test=y_te_f,
    )
    print("  Selesai menyimpan.")

    # ── 5. Verifikasi ─────────────────────────────────────────────────────
    print()
    verify_output(OUTPUT_NPZ)

    # ── 6. Ringkasan & panduan ────────────────────────────────────────────
    sep("═")
    print("  LABEL MAPPING (simpan ini untuk keperluan interpretasi hasil):")
    sep("─")
    print("  Label 0  =  non-fall  →  A008, A009, A027, A042")
    print("  Label 1  =  fall      →  A043")
    sep("─")
    print()
    print("  Langkah berikutnya — mulai training:")
    print()
    print("    cd ../../")
    print("    python main.py --config ./config/nturgbd-cross-subject/default.yaml")
    print()
    print("  Untuk memeriksa isi NPZ yang dihasilkan:")
    print()
    print("    python inspect_npz.py")
    sep("═")
