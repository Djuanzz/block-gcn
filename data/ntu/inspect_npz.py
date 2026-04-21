"""
inspect_npz.py
==============
Script untuk melihat isi file .npz hasil seq_transformation.py (BlockGCN/NTU60).
Taruh di folder mana saja, sesuaikan path NPZ_FILE di bawah.

Jalankan:
    python inspect_npz.py
"""

import numpy as np
import os

# =============================================================
#  EDIT INI — path ke file .npz yang ingin dilihat
# =============================================================
NPZ_FILE = "./NTU60_CS_filtered_5class.npz"
# =============================================================

NTU60_CLASSES = {
    1:"drink water", 2:"eat meal/snack", 3:"brushing teeth",
    4:"brushing hair", 5:"drop", 6:"pickup", 7:"throw",
    8:"sitting down", 9:"standing up", 10:"clapping",
    11:"reading", 12:"writing", 13:"tear up paper",
    14:"wear jacket", 15:"take off jacket", 16:"wear a shoe",
    17:"take off a shoe", 18:"wear on glasses", 19:"take off glasses",
    20:"put on a hat/cap", 21:"take off a hat/cap", 22:"cheer up",
    23:"hand waving", 24:"kicking something", 25:"reach into pocket",
    26:"hopping", 27:"jump up", 28:"make a phone call",
    29:"playing with phone/tablet", 30:"typing on a keyboard",
    31:"pointing to something", 32:"taking a selfie",
    33:"check time (from watch)", 34:"rub two hands together",
    35:"nod head/bow", 36:"shake head", 37:"wipe face",
    38:"salute", 39:"put the palms together", 40:"cross hands in front",
    41:"sneeze/cough", 42:"staggering", 43:"falling down",
    44:"touch head (headache)", 45:"touch chest (chest pain)",
    46:"touch back (backache)", 47:"touch neck (neckache)",
    48:"nausea or vomiting", 49:"use a fan / feeling warm",
    50:"punching/slapping other person", 51:"kicking other person",
    52:"pushing other person", 53:"pat on back of other person",
    54:"point finger at the other person", 55:"hugging other person",
    56:"giving something to other person", 57:"touch other person's pocket",
    58:"handshaking", 59:"walking towards each other",
    60:"walking apart from each other",
}

def sep(char="=", n=65):
    print(char * n)

def inspect(npz_path):
    if not os.path.exists(npz_path):
        print(f"\n[ERROR] File tidak ditemukan: {npz_path}")
        return

    print()
    sep()
    print(f"  FILE  : {os.path.abspath(npz_path)}")
    print(f"  SIZE  : {os.path.getsize(npz_path)/1024/1024:.2f} MB")
    sep()

    npz = np.load(npz_path)

    # ------------------------------------------------------------------
    # 1. Semua key yang ada
    # ------------------------------------------------------------------
    print(f"\n  KEYS dalam file ({len(npz.files)} key):")
    sep("-")
    for key in npz.files:
        arr = npz[key]
        print(f"  '{key}'")
        print(f"      shape  : {arr.shape}")
        print(f"      dtype  : {arr.dtype}")
        print(f"      min    : {arr.min():.4f}")
        print(f"      max    : {arr.max():.4f}")
        print(f"      mean   : {arr.mean():.4f}")
        print()

    # ------------------------------------------------------------------
    # 2. Penjelasan dimensi skeleton (x_train / x_test)
    # ------------------------------------------------------------------
    if 'x_train' in npz.files:
        x = npz['x_train']
        print()
        sep()
        print("  PENJELASAN DIMENSI x_train / x_test")
        sep()
        if x.ndim == 5:
            N, C, T, V, M = x.shape
            print(f"  Shape  : ({N}, {C}, {T}, {V}, {M})")
            print(f"  N = {N:>6}  → jumlah sample / sequence")
            print(f"  C = {C:>6}  → channel koordinat (biasanya 3: x, y, z)")
            print(f"  T = {T:>6}  → jumlah frame per sequence")
            print(f"  V = {V:>6}  → jumlah joint / keypoint per orang")
            print(f"  M = {M:>6}  → jumlah orang per frame (max)")
        elif x.ndim == 4:
            N, C, T, V = x.shape
            print(f"  Shape  : ({N}, {C}, {T}, {V})")
            print(f"  N = {N:>6}  → jumlah sample / sequence")
            print(f"  C = {C:>6}  → channel koordinat")
            print(f"  T = {T:>6}  → jumlah frame per sequence")
            print(f"  V = {V:>6}  → jumlah joint / keypoint")

    # ------------------------------------------------------------------
    # 3. Distribusi label per class
    # ------------------------------------------------------------------
    for split, lbl_key, data_key in [("TRAIN", "y_train", "x_train"),
                                      ("TEST",  "y_test",  "x_test")]:
        if lbl_key not in npz.files:
            continue

        y = npz[lbl_key]
        print()
        sep()
        print(f"  DISTRIBUSI LABEL — {split} SET")
        sep()

        # Deteksi format label: one-hot (2D) atau integer (1D)
        if y.ndim == 2:
            labels_int = np.argmax(y, axis=1)   # one-hot → integer 0-based
            fmt = "one-hot"
        else:
            labels_int = y.astype(int)
            fmt = "integer"

        num_classes_in_file = int(labels_int.max()) + 1
        total_samples = len(labels_int)

        print(f"  Format label  : {fmt}  (shape: {y.shape})")
        print(f"  Total sample  : {total_samples}")
        print(f"  Jumlah class  : {num_classes_in_file}")
        print()
        print(f"  {'No':<5} {'Class (1-based)':<20} {'Nama Action':<38} {'Jumlah':>7}")
        sep("-")

        unique, counts = np.unique(labels_int, return_counts=True)
        for cls_0based, count in zip(unique, counts):
            cls_1based = int(cls_0based) + 1
            name = NTU60_CLASSES.get(cls_1based, f"(class {cls_1based})")
            pct = count / total_samples * 100
            bar = "█" * int(pct / 2)
            print(f"  {cls_0based:<5} A{cls_1based:03d}                "
                  f"{name:<38} {count:>5}  ({pct:4.1f}%) {bar}")

    # ------------------------------------------------------------------
    # 4. Contoh sample pertama
    # ------------------------------------------------------------------
    if 'x_train' in npz.files and 'y_train' in npz.files:
        x = npz['x_train']
        y = npz['y_train']
        labels_int = np.argmax(y, axis=1) if y.ndim == 2 else y.astype(int)

        print()
        sep()
        print("  PREVIEW — 5 SAMPLE PERTAMA (x_train)")
        sep()
        print(f"  {'Idx':<6} {'Label (0-based)':<18} {'Class NTU (A-xxx)':<22} {'Nama Action'}")
        sep("-")
        for i in range(min(5, len(labels_int))):
            lbl = int(labels_int[i])
            cls_1based = lbl + 1
            name = NTU60_CLASSES.get(cls_1based, "Unknown")
            print(f"  {i:<6} {lbl:<18} A{cls_1based:03d}                   {name}")

        # Cek apakah ada nilai NaN atau Inf
        print()
        x_all = npz['x_train']
        nan_count = np.isnan(x_all).sum()
        inf_count = np.isinf(x_all).sum()
        print(f"  Cek kualitas data x_train:")
        print(f"    NaN values : {nan_count}")
        print(f"    Inf values : {inf_count}")
        if nan_count == 0 and inf_count == 0:
            print(f"    ✓ Data bersih, tidak ada NaN/Inf")

    print()
    sep()
    print("  SELESAI")
    sep()
    print()


if __name__ == "__main__":
    inspect(NPZ_FILE)