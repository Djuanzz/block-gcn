"""
inspect_npz.py
==============
Script untuk memeriksa isi dan kualitas file .npz dataset fall detection.

Jalankan (dari folder data/ntu/):
    python inspect_npz.py
    python inspect_npz.py NTU60_CS.npz          # periksa file tertentu

File yang dihasilkan prepare_fall_dataset.py:
    NTU60_CS_binary_fall.npz  —  2 kelas: 0=non-fall, 1=fall
"""

import sys
import numpy as np
import os

# =============================================================
#  Path default (bisa di-override lewat argumen command line)
# =============================================================
NPZ_FILE = sys.argv[1] if len(sys.argv) > 1 else "./NTU60_CS_binary_fall.npz"
# =============================================================

BINARY_LABEL_NAMES = {0: "non-fall", 1: "fall"}

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
    # Deteksi apakah ini file binary (2 kelas) untuk tampilan khusus
    is_binary = False
    if 'y_train' in npz.files:
        y_tmp = npz['y_train']
        n_cls = y_tmp.shape[1] if y_tmp.ndim == 2 else int(y_tmp.max()) + 1
        is_binary = (n_cls == 2)

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
            labels_int = np.argmax(y, axis=1)
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

        unique, counts = np.unique(labels_int, return_counts=True)

        if is_binary:
            # Tampilan khusus binary fall detection
            print(f"  {'Label':<8} {'Nama':<12} {'Jumlah':>8}  {'%':>6}  Bar")
            sep("-")
            for cls_0based, count in zip(unique, counts):
                name = BINARY_LABEL_NAMES.get(int(cls_0based), f"class_{cls_0based}")
                pct  = count / total_samples * 100
                bar  = "��" * int(pct / 2)
                print(f"  {cls_0based:<8} {name:<12} {count:>8}  {pct:>5.1f}%  {bar}")
            # Rasio imbalance
            if len(counts) == 2:
                ratio = counts[0] / counts[1] if counts[1] > 0 else float('inf')
                print(f"\n  Rasio non-fall/fall: {ratio:.1f} : 1"
                      f"  ({'imbalanced' if ratio > 3 else 'relatif seimbang'})")
        else:
            # Tampilan umum multi-class
            print(f"  {'No':<5} {'Class (1-based)':<20} {'Nama Action':<38} {'Jumlah':>7}")
            sep("-")
            for cls_0based, count in zip(unique, counts):
                cls_1based = int(cls_0based) + 1
                name = NTU60_CLASSES.get(cls_1based, f"(class {cls_1based})")
                pct  = count / total_samples * 100
                bar  = "█" * int(pct / 2)
                print(f"  {cls_0based:<5} A{cls_1based:03d}                "
                      f"{name:<38} {count:>5}  ({pct:4.1f}%) {bar}")

    # ------------------------------------------------------------------
    # 4. Cek kualitas data + preview
    # ------------------------------------------------------------------
    if 'x_train' in npz.files and 'y_train' in npz.files:
        x_all      = npz['x_train']
        y          = npz['y_train']
        labels_int = np.argmax(y, axis=1) if y.ndim == 2 else y.astype(int)

        print()
        sep()
        print("  KUALITAS DATA x_train")
        sep()
        nan_count = int(np.isnan(x_all).sum())
        inf_count = int(np.isinf(x_all).sum())
        zero_seq  = int((x_all.reshape(len(x_all), -1).sum(axis=1) == 0).sum())
        print(f"  NaN values    : {nan_count}")
        print(f"  Inf values    : {inf_count}")
        print(f"  Sequence nol  : {zero_seq}")
        if nan_count == 0 and inf_count == 0:
            print(f"  ✓ Data bersih, tidak ada NaN/Inf")

        if is_binary:
            print()
            sep()
            print("  PREVIEW — 5 SAMPLE x_train")
            sep()
            print(f"  {'Idx':<6} {'Label':<8} {'Kelas'}")
            sep("-")
            for i in range(min(5, len(labels_int))):
                lbl  = int(labels_int[i])
                name = BINARY_LABEL_NAMES.get(lbl, f"class_{lbl}")
                print(f"  {i:<6} {lbl:<8} {name}")

    print()
    sep()
    print("  SELESAI")
    sep()
    print()


if __name__ == "__main__":
    inspect(NPZ_FILE)