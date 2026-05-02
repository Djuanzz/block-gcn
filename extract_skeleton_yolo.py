"""
Ekstraksi skeleton YOLO11-pose dari video NTU RGB+D.

Langkah dalam pipeline persiapan data:
    1. extract_skeleton_yolo.py   ← script ini
    2. data/ntu_yolo/prepare_dataset_yolo.py

Naming convention video NTU RGB+D:
    S{setup:03d}C{camera:03d}P{performer:03d}R{rep:03d}A{action:03d}_rgb.avi
    Contoh: S001C001P001R001A043_rgb.avi

Output:
    <output_dir>/skeletons_raw.npz
        skeletons  : (N, T=300, 51)   float32
                     51 = 17 joint × 3 (x_norm, y_norm, confidence)
                     koordinat dinormalisasi ke [0,1] terhadap ukuran frame
        labels     : (N,) int32   — action class 1-indexed (8/9/27/42/43)
        subjects   : (N,) int32   — performer ID
        cameras    : (N,) int32   — camera ID
        setups     : (N,) int32   — setup ID
        filenames  : (N,) str     — nama file video asli

Contoh penggunaan:

Windows:
    python extract_skeleton_yolo.py ^
        --video-dir D:/NTU_RGB+D/nturgbd_rgb ^
        --output data/ntu_yolo/skeletons_raw.npz ^
        --classes 8 9 27 42 43 ^
        --model yolo11m-pose.pt ^
        --device 0

Google Colab:
    !python extract_skeleton_yolo.py \\
        --video-dir /content/drive/MyDrive/Fall-Detection/nturgbd_rgb \\
        --output /content/drive/MyDrive/Fall-Detection/dataset/yolo_pose/skeletons_raw.npz \\
        --classes 8 9 27 42 43 \\
        --model yolo11m-pose.pt \\
        --device 0

Dependencies:
    pip install ultralytics opencv-python tqdm
"""

import os
import re
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from ultralytics import YOLO
    import cv2
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(x, **kwargs):
        return x


# ── Split definisi ─────────────────────────────────────────────────────────
CROSS_SUBJECT_TRAIN_SUBJECTS = {
    1, 2, 4, 5, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
    25, 27, 28, 31, 34, 35, 38
}
CROSS_VIEW_TRAIN_CAMERAS = {2, 3}


def parse_ntu_filename(filename: str) -> Optional[dict]:
    """
    Parse nama file NTU RGB+D.
    Mendukung format: SsssXsssAsss_rgb.avi / SsssXsssAsss.avi (case-insensitive)
    Return dict {setup, camera, performer, replication, action} atau None jika tidak cocok.
    """
    stem = Path(filename).stem.replace('_rgb', '').replace('_RGB', '')
    m = re.match(r'S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})', stem, re.IGNORECASE)
    if not m:
        return None
    return {
        'setup':       int(m.group(1)),
        'camera':      int(m.group(2)),
        'performer':   int(m.group(3)),
        'replication': int(m.group(4)),
        'action':      int(m.group(5)),
    }


def select_main_person(result):
    """
    Pilih satu orang utama dari hasil deteksi YOLO.
    Strategi: ambil bounding box dengan area terbesar.
    Return: (xy, conf) masing-masing shape (17,2) dan (17,) atau None jika tidak ada deteksi.
    """
    boxes = result.boxes
    kps   = result.keypoints

    # Cek kps ada dan tidak kosong
    if kps is None:
        return None
    xy_tensor = kps.xy
    if xy_tensor is None or xy_tensor.shape[0] == 0:
        return None

    # Pilih orang dengan bbox terbesar; fallback ke indeks 0
    idx = 0
    if boxes is not None and len(boxes) > 0:
        try:
            areas = (boxes.xywh[:, 2] * boxes.xywh[:, 3]).cpu().numpy()
            idx = int(np.argmax(areas))
        except Exception:
            idx = 0

    # Pastikan idx tidak melebihi jumlah orang yang terdeteksi
    idx = min(idx, xy_tensor.shape[0] - 1)

    xy = xy_tensor[idx].cpu().numpy().astype(np.float32)  # (17, 2)

    # kps.conf bisa None pada beberapa konfigurasi YOLO
    if kps.conf is not None:
        conf = kps.conf[idx].cpu().numpy().astype(np.float32)  # (17,)
    else:
        conf = np.ones(17, dtype=np.float32)

    return xy, conf


def extract_video(video_path: Path, model, max_frames: int = 300) -> Optional[np.ndarray]:
    """
    Ekstrak keypoints dari satu file video.

    Return:
        (T=max_frames, 51) float32 — sudah di-pad/truncate
        x,y dinormalisasi ke [0,1] terhadap lebar/tinggi frame.
        None jika video tidak bisa dibuka.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_kps_list = []
    while len(frame_kps_list) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Lewati frame rusak
        if frame is None or frame.size == 0:
            frame_kps_list.append(np.zeros(51, dtype=np.float32))
            continue

        h, w = frame.shape[:2]

        try:
            results = model(frame, verbose=False)
            person  = select_main_person(results[0]) if results else None
        except Exception:
            person = None

        if person is not None:
            xy, conf = person
            xy_norm = xy.copy()
            xy_norm[:, 0] /= max(w, 1)
            xy_norm[:, 1] /= max(h, 1)
            # Klem ke [0,1] — koordinat tidak boleh negatif atau > 1
            xy_norm = np.clip(xy_norm, 0.0, 1.0)
            kp = np.concatenate([xy_norm, conf[:, None]], axis=1).flatten()  # (51,)
        else:
            kp = np.zeros(51, dtype=np.float32)

        frame_kps_list.append(kp)

    cap.release()

    if len(frame_kps_list) == 0:
        return None

    skeleton = np.array(frame_kps_list, dtype=np.float32)  # (T_actual, 51)

    # Pad jika kurang dari max_frames, truncate jika lebih
    T_actual = skeleton.shape[0]
    if T_actual < max_frames:
        pad = np.zeros((max_frames - T_actual, 51), dtype=np.float32)
        skeleton = np.concatenate([skeleton, pad], axis=0)
    else:
        skeleton = skeleton[:max_frames]

    return skeleton  # (300, 51)


def main():
    parser = argparse.ArgumentParser(
        description='Ekstrak skeleton YOLO11-pose dari video NTU RGB+D'
    )
    parser.add_argument('--video-dir', default="e:\\000 tugasakhir\\03 code\\Fall-Detection\\dataset\\ntu_videos",
                        help='Direktori berisi video NTU RGB+D (.avi / .mp4)')
    parser.add_argument('--output', default='data/ntu_yolo/skeletons_raw.npz',
                        help='Path output NPZ')
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[8, 9, 27, 42, 43],
                        help='Action class (1-indexed) yang diproses')
    parser.add_argument('--model', default='yolo11n-pose.pt',
                        help='Model YOLO pose: yolo11n-pose.pt / yolo11s-pose.pt / yolo11m-pose.pt')
    parser.add_argument('--max-frames', type=int, default=300,
                        help='Jumlah frame maksimum per video (default: 300)')
    parser.add_argument('--device', default='0',
                        help='GPU device: 0, 1, ... atau "cpu"')
    parser.add_argument('--split', default='cross-subject',
                        choices=['cross-subject', 'cross-view'],
                        help='Jenis split dataset (hanya untuk logging)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Lewati jika output sudah ada')
    args = parser.parse_args()

    if not DEPS_AVAILABLE:
        raise ImportError(
            'Dependensi tidak ditemukan. Install dengan:\n'
            '    pip install ultralytics opencv-python tqdm'
        )

    output_path = Path(args.output)
    if args.skip_existing and output_path.exists():
        print(f'Output sudah ada, dilewati: {output_path}')
        return

    target_classes = set(args.classes)
    print(f'Target action classes: {sorted(target_classes)}')
    print(f'Split: {args.split}')

    # ── Load model YOLO ──────────────────────────────────────────────────
    print(f'\nMemuat model YOLO: {args.model}')
    model = YOLO(args.model)
    device = args.device if args.device == 'cpu' else int(args.device)

    # ── Scan video ──────────────────────────────────────────────────────
    video_dir = Path(args.video_dir)
    print(f'Scanning: {video_dir}')
    all_videos = sorted(
        list(video_dir.rglob('*.avi')) + list(video_dir.rglob('*.mp4'))
    )
    print(f'Total video ditemukan: {len(all_videos)}')

    # Filter berdasarkan action class
    matched = []
    for vpath in all_videos:
        meta = parse_ntu_filename(vpath.name)
        if meta and meta['action'] in target_classes:
            matched.append((vpath, meta))

    print(f'Video cocok dengan class {sorted(target_classes)}: {len(matched)}')
    if len(matched) == 0:
        print(
            'Tidak ada video yang cocok!\n'
            'Pastikan --video-dir berisi file dengan format SsssXsssAsss_rgb.avi\n'
            'dan --classes sesuai dengan action ID yang ada.'
        )
        return

    # ── Ekstraksi ────────────────────────────────────────────────────────
    all_skeletons = []
    all_labels    = []
    all_subjects  = []
    all_cameras   = []
    all_setups    = []
    all_filenames = []
    skipped = 0

    for vpath, meta in tqdm(matched, desc='Ekstraksi skeleton'):
        skeleton = extract_video(vpath, model, max_frames=args.max_frames)
        if skeleton is None:
            print(f'  SKIP (gagal baca): {vpath.name}')
            skipped += 1
            continue

        all_skeletons.append(skeleton)
        all_labels.append(meta['action'])
        all_subjects.append(meta['performer'])
        all_cameras.append(meta['camera'])
        all_setups.append(meta['setup'])
        all_filenames.append(vpath.name)

    n_ok = len(all_skeletons)
    print(f'\nBerhasil: {n_ok} video  |  Dilewati: {skipped}')

    if n_ok == 0:
        print('Tidak ada skeleton yang diekstrak. Pipeline dihentikan.')
        return

    # Distribusi class
    unique_cls, counts = np.unique(all_labels, return_counts=True)
    print('Distribusi class:')
    for c, n in zip(unique_cls, counts):
        print(f'  A{c:03d}: {n} video')

    # ── Simpan NPZ ───────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        skeletons=np.array(all_skeletons, dtype=np.float32),   # (N, 300, 51)
        labels=np.array(all_labels,    dtype=np.int32),
        subjects=np.array(all_subjects, dtype=np.int32),
        cameras=np.array(all_cameras,   dtype=np.int32),
        setups=np.array(all_setups,     dtype=np.int32),
        filenames=np.array(all_filenames, dtype=object),
    )
    print(f'\nDisimpan ke: {output_path}')
    print(f'Shape skeletons: {np.array(all_skeletons).shape}')
    print('\nLangkah berikutnya:')
    print(f'  python data/ntu_yolo/prepare_dataset_yolo.py --input {output_path}')


if __name__ == '__main__':
    main()
