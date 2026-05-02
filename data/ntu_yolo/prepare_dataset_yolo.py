"""
Persiapan dataset binary fall/non-fall dari hasil ekstraksi YOLO11-pose.

Langkah dalam pipeline:
    1. extract_skeleton_yolo.py  → skeletons_raw.npz
    2. prepare_dataset_yolo.py   ← script ini → NTU60_YOLO_CS_binary_fall.npz

Label biner:
    0 = non-fall  : A008 (duduk), A009 (berdiri), A027 (melompat), A042 (limbung)
    1 = fall      : A043 (jatuh)

Output NPZ (kompatibel dengan feeders/feeder_yolo.py):
    x_train / x_test : (N, T=300, 51)  float32
    y_train / y_test : (N, 2)           float32  one-hot

Contoh penggunaan:

Windows:
    python data/ntu_yolo/prepare_dataset_yolo.py ^
        --input data/ntu_yolo/skeletons_raw.npz ^
        --output data/ntu_yolo/NTU60_YOLO_CS_binary_fall.npz

Google Colab:
    !python data/ntu_yolo/prepare_dataset_yolo.py \\
        --input /content/drive/MyDrive/Fall-Detection/dataset/yolo_pose/skeletons_raw.npz \\
        --output /content/drive/MyDrive/Fall-Detection/dataset/yolo_pose/NTU60_YOLO_CS_binary_fall.npz
"""

import os
import argparse
import numpy as np


# ── Mapping class ──────────────────────────────────────────────────────────
FALL_CLASSES     = frozenset({43})
NON_FALL_CLASSES = frozenset({8, 9, 27, 42})
ALL_CLASSES      = FALL_CLASSES | NON_FALL_CLASSES

# NTU60 cross-subject train — performer IDs
CROSS_SUBJECT_TRAIN = frozenset({
    1, 2, 4, 5, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
    25, 27, 28, 31, 34, 35, 38
})
# NTU60 cross-view train — camera IDs
CROSS_VIEW_TRAIN = frozenset({2, 3})


# ── Normalisasi ────────────────────────────────────────────────────────────

def normalize_skeletons(skeletons: np.ndarray) -> np.ndarray:
    """
    Normalisasi koordinat skeleton YOLO.

    Input  : (N, T, 51)   koordinat sudah [0,1] dari ukuran frame
    Output : (N, T, 51)   dicentrasi pada titik tengah pinggul,
                           diskalakan oleh jarak bahu-pinggul rata-rata

    Tujuan: menghilangkan posisi global (translasi) dan variasi skala tubuh,
    sesuai dengan preprocessing seq_transformation pada pipeline NTU asli.
    """
    N, T, D = skeletons.shape
    kps = skeletons.reshape(N, T, 17, 3)          # (N, T, 17, 3)
    xy   = kps[:, :, :, :2].copy()               # (N, T, 17, 2)
    conf = kps[:, :, :, 2:3].copy()              # (N, T, 17, 1)

    # ── 1. Centering pada titik tengah pinggul ──────────────────────────
    # left_hip=11, right_hip=12
    hip_center = (xy[:, :, 11:12, :] + xy[:, :, 12:13, :]) / 2.0  # (N,T,1,2)
    hip_conf   = (conf[:, :, 11, 0:1] + conf[:, :, 12, 0:1]) / 2.0  # (N,T,1)

    # Hanya geser frame yang pinggulnya terdeteksi (conf > threshold)
    hip_detected = (hip_conf > 0.1)[..., np.newaxis]  # (N,T,1,1)
    xy -= np.where(hip_detected, hip_center, 0.0)

    # ── 2. Normalisasi skala dengan jarak bahu kiri - pinggul kiri ──────
    left_shoulder = xy[:, :, 5, :]   # (N, T, 2)
    left_hip      = xy[:, :, 11, :]  # (N, T, 2)
    dist = np.linalg.norm(left_shoulder - left_hip, axis=-1)  # (N, T)

    # Rata-rata seluruh frame per sampel sebagai faktor skala
    mean_dist = dist.mean(axis=1, keepdims=True)             # (N, 1)
    mean_dist = np.where(mean_dist > 1e-4, mean_dist, 1.0)  # hindari div-by-zero

    xy /= mean_dist[:, :, np.newaxis, np.newaxis]            # broadcast (N,T,1,1)

    # ── Rekonstruksi ─────────────────────────────────────────────────────
    kps_norm = np.concatenate([xy, conf], axis=-1)           # (N, T, 17, 3)
    return kps_norm.reshape(N, T, 51).astype(np.float32)


def to_onehot(labels: np.ndarray, n_classes: int = 2) -> np.ndarray:
    oh = np.zeros((len(labels), n_classes), dtype=np.float32)
    oh[np.arange(len(labels)), labels] = 1.0
    return oh


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Buat dataset binary fall/non-fall dari skeleton YOLO'
    )
    parser.add_argument('--input',
                        default='data/ntu_yolo/skeletons_raw.npz',
                        help='NPZ output dari extract_skeleton_yolo.py')
    parser.add_argument('--output',
                        default='data/ntu_yolo/NTU60_YOLO_CS_binary_fall.npz',
                        help='Path output dataset binary')
    parser.add_argument('--split', default='cross-subject',
                        choices=['cross-subject', 'cross-view'],
                        help='Jenis split train/test')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Lewati normalisasi koordinat')
    args = parser.parse_args()

    # ── Load raw ──────────────────────────────────────────────────────────
    print(f'Memuat data dari: {args.input}')
    raw = np.load(args.input, allow_pickle=True)

    skeletons  = raw['skeletons']   # (N, T, 51)
    labels_raw = raw['labels']      # (N,) action class 1-indexed
    subjects   = raw['subjects']    # (N,) performer ID
    cameras    = raw['cameras']     # (N,) camera ID

    N_total = len(labels_raw)
    print(f'Total sampel raw: {N_total}')

    unique_cls, cls_counts = np.unique(labels_raw, return_counts=True)
    print('Distribusi class:')
    for c, n in zip(unique_cls, cls_counts):
        tag = 'FALL' if c in FALL_CLASSES else 'non-fall'
        print(f'  A{c:03d} ({tag}): {n}')

    # ── Filter class ──────────────────────────────────────────────────────
    mask = np.isin(labels_raw, list(ALL_CLASSES))
    skeletons  = skeletons[mask]
    labels_raw = labels_raw[mask]
    subjects   = subjects[mask]
    cameras    = cameras[mask]
    print(f'\nSetelah filter class {sorted(ALL_CLASSES)}: {len(labels_raw)} sampel')

    # ── Buat label biner ──────────────────────────────────────────────────
    binary_labels = np.isin(labels_raw, list(FALL_CLASSES)).astype(np.int32)

    # ── Split train / test ────────────────────────────────────────────────
    if args.split == 'cross-subject':
        train_mask = np.array([int(s) in CROSS_SUBJECT_TRAIN for s in subjects])
    else:
        train_mask = np.array([int(c) in CROSS_VIEW_TRAIN for c in cameras])
    test_mask = ~train_mask

    x_train = skeletons[train_mask]
    y_train_int = binary_labels[train_mask]
    x_test  = skeletons[test_mask]
    y_test_int  = binary_labels[test_mask]

    print(f'\nTrain : {len(x_train):5d} sampel  '
          f'(fall={y_train_int.sum()}, non-fall={(y_train_int==0).sum()})')
    print(f'Test  : {len(x_test):5d} sampel  '
          f'(fall={y_test_int.sum()}, non-fall={(y_test_int==0).sum()})')

    # ── Normalisasi ───────────────────────────────────────────────────────
    if not args.no_normalize:
        print('\nNormalisasi koordinat skeleton...')
        x_train = normalize_skeletons(x_train)
        x_test  = normalize_skeletons(x_test)
        print('  Selesai.')

    # ── One-hot encoding ──────────────────────────────────────────────────
    y_train = to_onehot(y_train_int)
    y_test  = to_onehot(y_test_int)

    # ── Simpan ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(
        args.output,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    print(f'\nDataset disimpan ke: {args.output}')
    print(f'  x_train : {x_train.shape}  dtype={x_train.dtype}')
    print(f'  y_train : {y_train.shape}')
    print(f'  x_test  : {x_test.shape}   dtype={x_test.dtype}')
    print(f'  y_test  : {y_test.shape}')
    print('\nLangkah berikutnya:')
    print('  python main.py --config config/yolo-pose-cross-subject/default.yaml')


if __name__ == '__main__':
    main()
