"""
filter_ntu_npz_binary.py
========================
Filter .npz NTU60 menjadi binary classification: fall vs not-fall.

Jalankan:
    python filter_ntu_npz_binary.py
"""

import numpy as np
import os

# =============================================================
#  CONFIG — EDIT SESUAI KEBUTUHAN
# =============================================================

INPUT_NPZ   = "./NTU60_CS.npz"
OUTPUT_NAME = "NTU60_CS_binary_fall.npz"

# Class yang masuk ke dataset (1-based NTU60 class number)
SELECTED_CLASSES = [8, 9, 27, 42, 43]

# Tentukan mana yang "fall" (label=1), sisanya otomatis jadi not-fall (label=0)
FALL_CLASSES = [43]   # A043 = falling down

# =============================================================

NTU60_CLASSES = {
    8:"sitting down", 9:"standing up", 27:"jump up",
    42:"staggering", 43:"falling down",
}

def sep(char="=", n=60): print(char * n)

def filter_binary(input_path, selected_classes, fall_classes, output_name):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"[ERROR] File tidak ditemukan: {input_path}")

    print(); sep()
    print(f"  Membaca: {input_path}")
    npz = np.load(input_path)

    x_train = npz['x_train']
    y_train = npz['y_train']
    x_test  = npz['x_test']
    y_test  = npz['y_test']

    print(f"  [Original] x_train: {x_train.shape}, x_test: {x_test.shape}")

    # Konversi one-hot → integer 0-based
    train_labels = np.argmax(y_train, axis=1) if y_train.ndim == 2 else y_train.astype(int)
    test_labels  = np.argmax(y_test,  axis=1) if y_test.ndim  == 2 else y_test.astype(int)

    # Index 0-based dari class yang dipilih dan fall class
    selected_0based = [c - 1 for c in selected_classes]
    fall_0based     = [c - 1 for c in fall_classes]

    # ------------------------------------------------------------------
    # Filter & beri binary label
    # ------------------------------------------------------------------
    def process_split(x, labels_int, split_name):
        mask    = np.isin(labels_int, selected_0based)
        x_filt  = x[mask]
        lbl_filt= labels_int[mask]

        # Binary label: 1 jika fall, 0 jika not-fall
        binary_lbl = np.array([1 if l in fall_0based else 0 for l in lbl_filt])

        # One-hot dengan 2 class
        y_binary = np.eye(2, dtype=y_train.dtype)[binary_lbl]

        # Logging distribusi
        n_fall     = (binary_lbl == 1).sum()
        n_notfall  = (binary_lbl == 0).sum()
        print(f"\n  [{split_name}] Total: {len(binary_lbl)} sample")
        print(f"    label 0 (not-fall) : {n_notfall} sample")
        for c in selected_classes:
            if c not in fall_classes:
                cnt = (lbl_filt == (c-1)).sum()
                print(f"             └─ A{c:03d} {NTU60_CLASSES.get(c,'?'):<20}: {cnt}")
        print(f"    label 1 (fall)     : {n_fall} sample")
        for c in fall_classes:
            cnt = (lbl_filt == (c-1)).sum()
            print(f"             └─ A{c:03d} {NTU60_CLASSES.get(c,'?'):<20}: {cnt}")

        return x_filt, y_binary

    sep()
    print("  HASIL FILTERING + BINARY LABELING")
    sep()

    x_train_f, y_train_f = process_split(x_train, train_labels, "TRAIN")
    x_test_f,  y_test_f  = process_split(x_test,  test_labels,  "TEST")

    # ------------------------------------------------------------------
    # Simpan
    # ------------------------------------------------------------------
    out_dir  = os.path.dirname(os.path.abspath(input_path))
    out_path = os.path.join(out_dir, output_name)
    np.savez_compressed(
        out_path,
        x_train=x_train_f, y_train=y_train_f,
        x_test=x_test_f,   y_test=y_test_f,
    )

    sep()
    print(f"  [SUKSES] Tersimpan: {out_path}")
    print(f"  Ukuran : {os.path.getsize(out_path)/1024/1024:.1f} MB")
    sep()

    # ------------------------------------------------------------------
    # Panduan langkah selanjutnya
    # ------------------------------------------------------------------
    print(f"""
  LANGKAH SELANJUTNYA:

  1. config YAML  →  ubah:
       num_class: 2
       train_feeder_args:
         data_path: data/ntu/{output_name}
       test_feeder_args:
         data_path: data/ntu/{output_name}

  2. Label mapping yang berlaku:
       Label 0  =  not-fall  ({', '.join([f'A{c:03d}' for c in selected_classes if c not in fall_classes])})
       Label 1  =  fall      ({', '.join([f'A{c:03d}' for c in fall_classes])})
""")

if __name__ == "__main__":
    filter_binary(INPUT_NPZ, SELECTED_CLASSES, FALL_CLASSES, OUTPUT_NAME)