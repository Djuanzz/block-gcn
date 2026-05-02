"""
data/ntu/prepare_dataset_ntu_v2.py
====================================
Siapkan dataset NTU RGB+D 25-joint untuk training dengan metodologi Proyek B.

INPUT:
  NTU60_CS_binary_fall.npz  (sudah difilter ke 2 kelas oleh prepare_fall_dataset.py)
  Format: x_train/x_test (N, T=300, 150)  —  150 = M=2 × V=25 × C=3
          y_train/y_test (N, 2)  —  one-hot [non-fall, fall]

OUTPUT (folder yang sama dengan input):
  train_data.npy     (N_train, C=3, T=150, V=25, M=1)  float32
  train_label.pkl    (sample_names, labels)
  val_data.npy       (N_val, C=3, T=150, V=25, M=1)
  val_label.pkl
  dataset_info.json

METODOLOGI (identik dengan Proyek B / Fall-Detection):
  - Pilih orang dengan frame valid terbanyak (M=1)
  - Centering per frame pada SpineShoulder (joint 20)
  - Scale oleh mean lebar bahu kiri-kanan (joint 4 - joint 8)
  - Stratified random split 80/20 (seed=42)
  - Pad dengan nol / crop tengah ke max_frames=150

CARA PAKAI:
  cd block-gcn-yolo/
  python data/ntu/prepare_dataset_ntu_v2.py \\
      --input data/ntu/NTU60_CS_binary_fall.npz \\
      --output data/ntu \\
      --max_frames 150
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# ── Konstanta joint NTU (0-indexed) ─────────────────────────────────────────
SPINE_SHOULDER = 20   # titik pusat untuk centering
LEFT_SHOULDER  = 4
RIGHT_SHOULDER = 8


def parse_args():
    ap = argparse.ArgumentParser(
        description="Siapkan dataset NTU 25-joint dengan metodologi Proyek B")
    ap.add_argument("--input",      required=True,
                    help="Path ke NTU60_CS_binary_fall.npz")
    ap.add_argument("--output",     required=True,
                    help="Folder output untuk train/val .npy dan .pkl")
    ap.add_argument("--max_frames", type=int,   default=150,
                    help="Panjang sekuens output (default: 150)")
    ap.add_argument("--val_ratio",  type=float, default=0.2,
                    help="Proporsi data validasi (default: 0.2)")
    ap.add_argument("--seed",       type=int,   default=42)
    return ap.parse_args()


# ── Step 1: Baca dan gabungkan train+test ────────────────────────────────────

def load_and_merge(npz_path: str):
    """
    Gabungkan split train dan test dari NPZ binary fall.
    Return: x_all (N, T, 150), y_all (N,) integer label
    """
    d     = np.load(npz_path)
    x_all = np.concatenate([d['x_train'], d['x_test']], axis=0)
    y_all = np.concatenate([
        np.where(d['y_train'] > 0)[1],
        np.where(d['y_test']  > 0)[1],
    ], axis=0)
    return x_all.astype(np.float32), y_all.astype(np.int64)


# ── Step 2: Pilih orang terbaik (M=1) ───────────────────────────────────────

def select_best_person(x: np.ndarray) -> np.ndarray:
    """
    Dari 2 orang, pilih yang memiliki jumlah frame valid terbanyak.
    Input:  (N, T, 150)  — 150 = M=2 × V=25 × C=3
    Output: (N, T, 25, 3) — satu orang per sample
    """
    N, T, _ = x.shape
    data    = x.reshape(N, T, 2, 25, 3)   # (N, T, M, V, C)
    result  = np.zeros((N, T, 25, 3), dtype=np.float32)

    for i in range(N):
        valid_0 = int((data[i, :, 0].sum(axis=(1, 2)) != 0).sum())
        valid_1 = int((data[i, :, 1].sum(axis=(1, 2)) != 0).sum())
        m       = 0 if valid_0 >= valid_1 else 1
        result[i] = data[i, :, m]

    return result   # (N, T, 25, 3)


# ── Step 3: Normalisasi skeleton ─────────────────────────────────────────────

def normalize_skeleton(sk: np.ndarray) -> np.ndarray:
    """
    Normalisasi satu sample skeleton NTU.
    Input/output: (T, 25, 3)

    1. Centering per frame: translasi agar SpineShoulder (joint 20) di origin
    2. Scale: bagi dengan mean jarak bahu kiri-kanan dari seluruh sequence
    """
    sk  = sk.copy()
    xy  = sk[:, :, :2]   # (T, 25, 2) — hanya X dan Y untuk normalisasi skala

    # Centering: tiap frame dipusatkan ke SpineShoulder
    center = xy[:, SPINE_SHOULDER: SPINE_SHOULDER + 1, :]   # (T, 1, 2)
    sk[:, :, :2] -= center

    # Scale oleh lebar bahu rata-rata sepanjang sequence
    xy_centered = sk[:, :, :2]
    d  = np.linalg.norm(
        xy_centered[:, LEFT_SHOULDER] - xy_centered[:, RIGHT_SHOULDER],
        axis=1
    )   # (T,)
    sc = d[d > 1e-5].mean() if (d > 1e-5).any() else 1.0
    if sc > 1e-5:
        sk[:, :, :2] /= sc

    return sk


# ── Step 4: Pad atau crop ke max_frames ──────────────────────────────────────

def pad_or_crop(sk: np.ndarray, max_frames: int) -> np.ndarray:
    """
    Sesuaikan panjang temporal ke max_frames.
    - Lebih panjang: crop dari tengah
    - Lebih pendek: pad nol di akhir
    Input/output: (T, 25, 3)
    """
    T = sk.shape[0]
    if T >= max_frames:
        start = (T - max_frames) // 2
        return sk[start: start + max_frames].copy()
    pad = np.zeros((max_frames - T, 25, 3), dtype=np.float32)
    return np.concatenate([sk, pad], axis=0)


# ── Step 5: Konversi ke format tensor ────────────────────────────────────────

def to_tensor_format(sk: np.ndarray) -> np.ndarray:
    """
    (T, 25, 3) → (C=3, T=max_frames, V=25, M=1)
    """
    # (T, 25, 3) → transpose → (3, T, 25) → unsqueeze M → (3, T, 25, 1)
    return sk.transpose(2, 0, 1)[:, :, :, np.newaxis]


# ── Step 6: Simpan split ─────────────────────────────────────────────────────

def save_split(data: np.ndarray, labels: list, names: list,
               out_dir: Path, prefix: str):
    data_path  = out_dir / f"{prefix}_data.npy"
    label_path = out_dir / f"{prefix}_label.pkl"
    np.save(str(data_path), data)
    with open(str(label_path), "wb") as f:
        pickle.dump((names, labels), f)
    n_fall    = sum(labels)
    n_nonfal  = len(labels) - n_fall
    print(f"  [{prefix}] {len(labels)} samples "
          f"(fall={n_fall}, not_fall={n_nonfal})  → {data.shape}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Prepare NTU 25-joint Dataset  (Metodologi Proyek B)")
    print("=" * 60)

    # 1. Load
    print(f"\n[1] Memuat: {args.input}")
    x_raw, y_all = load_and_merge(args.input)
    print(f"    Total  : {len(y_all)}  "
          f"(fall={int((y_all==1).sum())}, not_fall={int((y_all==0).sum())})")

    # 2. Pilih orang terbaik
    print("[2] Memilih orang terbaik per sample (M=1)...")
    x_person = select_best_person(x_raw)   # (N, T, 25, 3)

    # 3. Normalisasi
    print("[3] Normalisasi (SpineShoulder centering + shoulder-width scale)...")
    x_norm = np.stack([
        normalize_skeleton(x_person[i]) for i in range(len(x_person))
    ], axis=0)   # (N, T, 25, 3)

    # 4. Pad/crop ke max_frames
    print(f"[4] Pad/crop ke {args.max_frames} frame...")
    x_padded = np.stack([
        pad_or_crop(x_norm[i], args.max_frames) for i in range(len(x_norm))
    ], axis=0)   # (N, max_frames, 25, 3)

    # 5. Konversi ke format tensor (N, C=3, T, V=25, M=1)
    print("[5] Konversi ke format (C, T, V, M)...")
    x_tensor = np.stack([
        to_tensor_format(x_padded[i]) for i in range(len(x_padded))
    ], axis=0).astype(np.float32)   # (N, 3, max_frames, 25, 1)

    # 6. Stratified split
    print(f"[6] Stratified split 80/20 (seed={args.seed})...")
    idx_all = np.arange(len(y_all))
    idx_tr, idx_val = train_test_split(
        idx_all,
        test_size    = args.val_ratio,
        stratify     = y_all,
        random_state = args.seed,
    )

    x_train = x_tensor[idx_tr]
    y_train = y_all[idx_tr].tolist()
    x_val   = x_tensor[idx_val]
    y_val   = y_all[idx_val].tolist()

    names_train = [f"train_{i}" for i in range(len(y_train))]
    names_val   = [f"val_{i}"   for i in range(len(y_val))]

    # 7. Simpan
    print("[7] Menyimpan dataset...")
    save_split(x_train, y_train, names_train, out_dir, "train")
    save_split(x_val,   y_val,   names_val,   out_dir, "val")

    # 8. Simpan info
    info = {
        "source_npz"     : str(args.input),
        "total_samples"  : int(len(y_all)),
        "train_samples"  : int(len(y_train)),
        "val_samples"    : int(len(y_val)),
        "fall_train"     : int(sum(y_train)),
        "fall_val"       : int(sum(y_val)),
        "not_fall_train" : int(len(y_train) - sum(y_train)),
        "not_fall_val"   : int(len(y_val)   - sum(y_val)),
        "max_frames"     : args.max_frames,
        "num_joints"     : 25,
        "channels"       : 3,
        "center_joint"   : SPINE_SHOULDER,
        "center_name"    : "SpineShoulder",
        "scale_joints"   : [LEFT_SHOULDER, RIGHT_SHOULDER],
        "scale_names"    : ["ShoulderLeft", "ShoulderRight"],
        "split_ratio"    : f"{int((1-args.val_ratio)*100)}/{int(args.val_ratio*100)}",
        "seed"           : args.seed,
        "tensor_shape"   : "(N, C=3, T, V=25, M=1)",
    }
    with open(str(out_dir / "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Selesai. Output: {out_dir}")
    print(f"  Train : {len(y_train)} "
          f"(fall={sum(y_train)}, not_fall={len(y_train)-sum(y_train)})")
    print(f"  Val   : {len(y_val)} "
          f"(fall={sum(y_val)}, not_fall={len(y_val)-sum(y_val)})")
    print("=" * 60)
    print("\nLangkah berikutnya — mulai training:")
    print("  python main.py --config config/ntu-25joint/default.yaml")
    print()


if __name__ == "__main__":
    main()
