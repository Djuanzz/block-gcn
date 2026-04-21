"""
filter_ntu_npz.py
=================
Script untuk mem-filter file .npz hasil seq_transformation.py (BlockGCN/CTR-GCN)
agar hanya berisi class-class tertentu yang kamu pilih.

CARA PAKAI:
-----------
1. Taruh script ini di folder: BlockGCN/data/ntu/
2. Edit bagian CONFIG di bawah sesuai kebutuhanmu
3. Jalankan:
       python filter_ntu_npz.py

OUTPUT:
-------
File .npz baru akan tersimpan di folder yang sama dengan nama custom,
contoh: NTU60_CS_filtered_3class.npz

Struktur .npz yang dihasilkan sama persis dengan aslinya:
  - x_train : array skeleton training  (N_train, C, T, V, M)
  - y_train : label training           (N_train, num_class)  ← one-hot
  - x_test  : array skeleton test      (N_test,  C, T, V, M)
  - y_test  : label test               (N_test,  num_class)  ← one-hot
"""

import numpy as np
import os

# =============================================================================
# =====================  EDIT BAGIAN INI SESUAI KEBUTUHANMU  ==================
# =============================================================================

# Path ke file .npz original hasil seq_transformation.py
# Sesuaikan apakah kamu pakai xsub (cross-subject) atau xview (cross-view)
INPUT_NPZ = "./NTU60_CS.npz"       # cross-subject
# INPUT_NPZ = "./NTU60_CV.npz"     # cross-view (uncomment jika pakai ini)

# Pilih class yang ingin kamu masukkan (gunakan nomor 1-based, sesuai label NTU60)
# Contoh di bawah: class untuk fall detection
#   A007 = hand waving
#   A008 = sitting down
#   A009 = standing up (from sitting position)
#   A043 = falling down  ← class utama untuk fall detection
#   A044 = headache
#   A045 = chest pain
SELECTED_CLASSES = [8, 9, 27, 42, 43]

# Nama file output (akan disimpan di folder yang sama dengan INPUT_NPZ)
OUTPUT_NAME = f"NTU60_CS_filtered_{len(SELECTED_CLASSES)}class.npz"

# =============================================================================
# ========================  JANGAN EDIT DI BAWAH INI  ========================
# =============================================================================

# Daftar semua 60 nama class NTU60 (untuk logging dan verifikasi)
NTU60_CLASSES = {
    1: "drink water", 2: "eat meal/snack", 3: "brushing teeth",
    4: "brushing hair", 5: "drop", 6: "pickup", 7: "throw",
    8: "sitting down", 9: "standing up (from sitting position)",
    10: "clapping", 11: "reading", 12: "writing", 13: "tear up paper",
    14: "wear jacket", 15: "take off jacket", 16: "wear a shoe",
    17: "take off a shoe", 18: "wear on glasses", 19: "take off glasses",
    20: "put on a hat/cap", 21: "take off a hat/cap", 22: "cheer up",
    23: "hand waving", 24: "kicking something", 25: "reach into pocket",
    26: "hopping (one foot jumping)", 27: "jump up", 28: "make a phone call/answer phone",
    29: "playing with phone/tablet", 30: "typing on a keyboard",
    31: "pointing to something with finger", 32: "taking a selfie",
    33: "check time (from watch)", 34: "rub two hands together",
    35: "nod head/bow", 36: "shake head", 37: "wipe face",
    38: "salute", 39: "put the palms together", 40: "cross hands in front (say stop)",
    41: "sneeze/cough", 42: "staggering", 43: "falling down",
    44: "touch head (headache)", 45: "touch chest (chest pain/heart pain)",
    46: "touch back (backache)", 47: "touch neck (neckache)",
    48: "nausea or vomiting condition", 49: "use a fan (with hand or paper)/feeling warm",
    50: "punching/slapping other person", 51: "kicking other person",
    52: "pushing other person", 53: "pat on back of other person",
    54: "point finger at the other person", 55: "hugging other person",
    56: "giving something to other person", 57: "touch other person's pocket",
    58: "handshaking", 59: "walking towards each other",
    60: "walking apart from each other",
}


def filter_npz(input_path, selected_classes, output_name):
    # -------------------------------------------------------------------------
    # 1. Validasi input
    # -------------------------------------------------------------------------
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"\n[ERROR] File tidak ditemukan: {input_path}\n"
            f"Pastikan kamu sudah menjalankan seq_transformation.py terlebih dahulu,\n"
            f"dan path INPUT_NPZ sudah benar.\n"
        )

    # -------------------------------------------------------------------------
    # 2. Load .npz original
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Membaca file: {input_path}")
    npz = np.load(input_path)
    print(f"  Keys dalam file: {list(npz.files)}")

    x_train = npz['x_train']   # shape: (N_train, C, T, V, M)
    y_train = npz['y_train']   # shape: (N_train, 60)  ← one-hot encoded
    x_test  = npz['x_test']    # shape: (N_test,  C, T, V, M)
    y_test  = npz['y_test']    # shape: (N_test,  60)  ← one-hot encoded

    print(f"\n  [Original] x_train shape : {x_train.shape}")
    print(f"  [Original] y_train shape : {y_train.shape}")
    print(f"  [Original] x_test  shape : {x_test.shape}")
    print(f"  [Original] y_test  shape : {y_test.shape}")

    # -------------------------------------------------------------------------
    # 3. Konversi one-hot → label integer (0-based)
    # -------------------------------------------------------------------------
    # y_train aslinya adalah one-hot dengan shape (N, 60)
    # np.argmax mengubahnya jadi integer: 0 s/d 59
    train_labels_int = np.argmax(y_train, axis=1)   # 0-based
    test_labels_int  = np.argmax(y_test,  axis=1)   # 0-based

    # -------------------------------------------------------------------------
    # 4. Buat mapping: class number (1-based) → index 0-based di .npz asli
    #    dan → new label (0-based, mulai dari 0 untuk class pertama yang dipilih)
    # -------------------------------------------------------------------------
    # Urutkan selected_classes supaya label baru konsisten
    selected_sorted = sorted(selected_classes)

    # Map: original_0based_label → new_0based_label
    orig_to_new = {}
    for new_idx, cls_1based in enumerate(selected_sorted):
        orig_0based = cls_1based - 1
        orig_to_new[orig_0based] = new_idx

    # -------------------------------------------------------------------------
    # 5. Filter training data
    # -------------------------------------------------------------------------
    train_mask = np.isin(train_labels_int, list(orig_to_new.keys()))
    x_train_filtered = x_train[train_mask]
    labels_train_filtered = train_labels_int[train_mask]

    # Remap ke label baru (0-based sesuai urutan selected_classes)
    labels_train_new = np.array([orig_to_new[l] for l in labels_train_filtered])

    # Buat one-hot baru dengan num_class = len(selected_classes)
    num_class = len(selected_sorted)
    y_train_new = np.eye(num_class, dtype=y_train.dtype)[labels_train_new]

    # -------------------------------------------------------------------------
    # 6. Filter test data
    # -------------------------------------------------------------------------
    test_mask = np.isin(test_labels_int, list(orig_to_new.keys()))
    x_test_filtered = x_test[test_mask]
    labels_test_filtered = test_labels_int[test_mask]

    labels_test_new = np.array([orig_to_new[l] for l in labels_test_filtered])
    y_test_new = np.eye(num_class, dtype=y_test.dtype)[labels_test_new]

    # -------------------------------------------------------------------------
    # 7. Logging ringkasan
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Class yang dipilih ({num_class} class):")
    for new_idx, cls_1based in enumerate(selected_sorted):
        cls_name = NTU60_CLASSES.get(cls_1based, "Unknown")
        orig_train_count = np.sum(train_labels_int == (cls_1based - 1))
        orig_test_count  = np.sum(test_labels_int  == (cls_1based - 1))
        print(f"    new_label={new_idx}  ← A{cls_1based:03d} ({cls_name})"
              f"  |  train={orig_train_count}, test={orig_test_count}")

    print(f"\n  [Filtered] x_train shape : {x_train_filtered.shape}")
    print(f"  [Filtered] y_train shape : {y_train_new.shape}")
    print(f"  [Filtered] x_test  shape : {x_test_filtered.shape}")
    print(f"  [Filtered] y_test  shape : {y_test_new.shape}")

    # -------------------------------------------------------------------------
    # 8. Simpan ke file .npz baru
    # -------------------------------------------------------------------------
    output_dir = os.path.dirname(os.path.abspath(input_path))
    output_path = os.path.join(output_dir, output_name)

    np.savez_compressed(
        output_path,
        x_train = x_train_filtered,
        y_train = y_train_new,
        x_test  = x_test_filtered,
        y_test  = y_test_new,
    )

    print(f"\n{'='*60}")
    print(f"  [SUKSES] File tersimpan: {output_path}")
    print(f"  Ukuran file: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    print(f"{'='*60}\n")

    return output_path, num_class


def print_label_mapping(selected_classes):
    """Helper untuk print mapping label — berguna saat konfigurasi training."""
    selected_sorted = sorted(selected_classes)
    print("\n  LABEL MAPPING (gunakan ini di config YAML dan inference):")
    print("  " + "-"*50)
    for new_idx, cls_1based in enumerate(selected_sorted):
        cls_name = NTU60_CLASSES.get(cls_1based, "Unknown")
        print(f"    Label {new_idx} = A{cls_1based:03d} = {cls_name}")
    print()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  BlockGCN / CTR-GCN — NTU60 Class Filter Script")
    print("="*60)

    # Print mapping dulu supaya user tau label barunya apa
    print_label_mapping(SELECTED_CLASSES)

    # Jalankan filter
    out_path, num_class = filter_npz(
        input_path      = INPUT_NPZ,
        selected_classes= SELECTED_CLASSES,
        output_name     = OUTPUT_NAME,
    )

    # -------------------------------------------------------------------------
    # 9. Print panduan langkah selanjutnya
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("  LANGKAH SELANJUTNYA SETELAH SCRIPT INI SELESAI:")
    print("="*60)
    print(f"""
  1. Di config YAML (contoh: config/nturgbd-cross-subject/default.yaml),
     ubah bagian ini:

       num_class: {num_class}   # ← sesuaikan dengan jumlah class yang dipilih

       train_feeder_args:
         data_path: data/ntu/{os.path.basename(out_path)}   # ← file output di atas
         ...

       test_feeder_args:
         data_path: data/ntu/{os.path.basename(out_path)}   # ← sama
         ...

  2. Di feeders/feeder_ntu.py, pastikan parameter 'num_class' di config
     diteruskan ke model, BUKAN hardcoded 60.
     (Biasanya ini sudah otomatis karena config YAML mengatur num_class)

  3. Jalankan training seperti biasa:
       bash train.sh

  NOTE: Label mapping baru (untuk inference nanti):
""")
    for new_idx, cls_1based in enumerate(sorted(SELECTED_CLASSES)):
        cls_name = NTU60_CLASSES.get(cls_1based, "Unknown")
        print(f"    Label {new_idx} → A{cls_1based:03d} = {cls_name}")
    print()