"""
predict.py  –  BlockGCN fall detection: batch evaluation with TXT report.

Menggunakan Feeder + DataLoader yang SAMA dengan training agar konsisten.

Cara pakai tercepat  →  edit bagian DEFAULT CONFIG di bawah, lalu:
  python predict.py

Atau override lewat argumen:
  python predict.py --skeleton-path X --weights Y --split test --output-txt Z
"""

import os
import sys
import argparse
import yaml
from collections import OrderedDict
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT CONFIG  —  edit bagian ini sesuai kebutuhan
# ═══════════════════════════════════════════════════════════════════════════════

# Path ke file dataset NPZ
SKELETON_PATH = "data/ntu/NTU60_CS_binary_fall.npz"

# Path ke file bobot model (.pt)
WEIGHTS_PATH  = "weights/2/runs-62-25916.pt"

# Path ke file konfigurasi YAML
CONFIG_PATH   = "config/nturgbd-cross-subject/default.yaml"

# Split yang dievaluasi: "test" atau "train"
SPLIT         = "test"

# Path output laporan TXT
OUTPUT_TXT    = "results/evaluation_report.txt"

# Folder statistik NTU (untuk nama file asli skeleton)
STATS_DIR     = "data/ntu/statistics"

# Device: "auto" (GPU jika tersedia), "cuda", atau "cpu"
DEVICE        = "auto"

# Batch size — harus SAMA dengan test_batch_size saat training
BATCH_SIZE    = 8

# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from feeders.feeder_ntu import Feeder


# ─── constants ──────────────────────────────────────────────────────────────────

FALL_LABEL   = 1
NFALL_LABEL  = 0
LABEL_STR    = {FALL_LABEL: "fall", NFALL_LABEL: "not_fall"}

_DIV_LONG  = "─" * 80
_DIV_SHORT = "─" * 50

# NTU CS split performer IDs — sama persis dengan seq_transformation.py
_TRAIN_IDS = [1,  2,  4,  5,  8,  9,  13, 14, 15, 16,
              17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
_TEST_IDS  = [3,  6,  7,  10, 11, 12, 20, 21, 22, 23,
              24, 26, 29, 30, 32, 33, 36, 37, 39, 40]

# Kelas NTU yang dipilih (1-based), fall class = 43
_SELECTED_1B = {8, 9, 27, 42, 43}


# ─── helpers ────────────────────────────────────────────────────────────────────

def import_class(name: str):
    import traceback
    mod_str, _, cls_str = name.rpartition(".")
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], cls_str)
    except AttributeError:
        raise ImportError("Class %s cannot be found (%s)" % (cls_str, traceback.format_exception(*sys.exc_info())))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


# ─── nama sampel asli ────────────────────────────────────────────────────────────

def reconstruct_sample_names(stats_dir: str, split: str, n_expected: int) -> list:
    """
    Rekonstruksi nama file skeleton asli (mis. S001C001P003R001A043)
    dengan meniru urutan filtering yang sama persis seperti:
      seq_transformation.py → get_indices()
      prepare_fall_dataset.py → filter_to_binary()

    Mengembalikan list nama dengan panjang n_expected,
    atau list fallback "sample_XXXX" jika stats tidak tersedia.
    """
    try:
        names     = np.loadtxt(os.path.join(stats_dir, "skes_available_name.txt"), dtype=str)
        performer = np.loadtxt(os.path.join(stats_dir, "performer.txt"),            dtype=int)
        label_1b  = np.loadtxt(os.path.join(stats_dir, "label.txt"),               dtype=int)
    except Exception as e:
        print(f"  [nama] Tidak bisa baca stats: {e}  → pakai nama sekuensial")
        return [f"sample_{i:04d}" for i in range(n_expected)]

    subject_ids = _TRAIN_IDS if split == "train" else _TEST_IDS

    # Replikasi get_indices: iterasi subject_ids in order, kumpulkan indeks global
    split_indices = []
    for pid in subject_ids:
        split_indices.extend(np.where(performer == pid)[0].tolist())
    split_indices = np.array(split_indices, dtype=int)

    # Replikasi filter_to_binary: mask ke kelas yang dipilih
    split_labels = label_1b[split_indices]
    mask         = np.isin(split_labels, list(_SELECTED_1B))
    filtered_idx = split_indices[mask]

    result = [str(names[i]) for i in filtered_idx]

    if len(result) != n_expected:
        print(f"  [nama] Rekonstruksi menghasilkan {len(result)} nama, "
              f"NPZ punya {n_expected} sampel → pakai nama sekuensial")
        return [f"sample_{i:04d}" for i in range(n_expected)]

    return result


# ─── model ──────────────────────────────────────────────────────────────────────

def load_model(cfg: dict, weights_path: str, device: torch.device):
    Model = import_class(cfg["model"])
    model = Model(**cfg["model_args"])
    model.to(device)

    raw = torch.load(weights_path, map_location=device)
    state = OrderedDict((k.replace("module.", ""), v) for k, v in raw.items())
    try:
        model.load_state_dict(state)
    except RuntimeError:
        ms = model.state_dict()
        ms.update({k: v for k, v in state.items() if k in ms})
        model.load_state_dict(ms)

    model.eval()
    return model


# ─── TXT report ─────────────────────────────────────────────────────────────────

def write_report(filepath: str, rows: list, labels: np.ndarray, preds: np.ndarray):
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    acc  = (tp + tn) / (tp + tn + fp + fn) * 100
    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    n_total = len(labels)
    n_benar = int((preds == labels).sum())
    n_salah = n_total - n_benar
    n_fall  = int((labels == FALL_LABEL).sum())
    n_nfall = int((labels == NFALL_LABEL).sum())

    hdr = (
        f"{'Nama Sampel':<26} | {'Ground Truth':<10} | {'Prediksi':<10} | "
        f"{'Status':<8} | Fall Prob"
    )
    sep = f"{'-'*26}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-----------"

    lines = []
    lines.append(f"# Hasil evaluasi BlockGCN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"# Split: {' '.join(sys.argv)}")
    lines.append("#")
    lines.append(hdr)
    lines.append(sep)

    for name, truth, pred, status, prob in rows:
        lines.append(f"{name:<26} | {truth:<10} | {pred:<10} | {status:<8} | {prob:.4f}")

    lines.append("")
    lines.append(f"# {_DIV_LONG}")
    lines.append("# RINGKASAN")
    lines.append(f"# {_DIV_LONG}")
    lines.append(f"# Total sampel  : {n_total}")
    lines.append(f"# Benar         : {n_benar}  ({n_benar/n_total*100:.2f}%)")
    lines.append(f"# Salah         : {n_salah}")
    lines.append(f"# {_DIV_SHORT}")
    lines.append(f"# Accuracy      : {acc:.2f}%")
    lines.append(f"# Precision     : {prec:.2f}%")
    lines.append(f"# Recall        : {rec:.2f}%")
    lines.append(f"# F1-Score      : {f1:.2f}%")
    lines.append(f"# {_DIV_SHORT}")
    lines.append("# Confusion Matrix:")
    lines.append(f"#   TN (not_fall benar)  : {tn}")
    lines.append(f"#   FP (false alarm)     : {fp}")
    lines.append(f"#   FN (missed fall)     : {fn}")
    lines.append(f"#   TP (fall terdeteksi) : {tp}")
    lines.append(f"# {_DIV_SHORT}")
    lines.append(f"# Missed fall  : {fn}/{n_fall} = {fn/n_fall*100:.1f}%" if n_fall > 0 else "# Missed fall  : N/A")
    lines.append(f"# False alarm  : {fp}/{n_nfall} = {fp/n_nfall*100:.1f}%" if n_nfall > 0 else "# False alarm  : N/A")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return acc, prec, rec, f1, tn, fp, fn, tp


# ─── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BlockGCN fall detection – batch evaluation")
    p.add_argument("--skeleton-path", default=SKELETON_PATH)
    p.add_argument("--weights",        default=WEIGHTS_PATH)
    p.add_argument("--config",         default=CONFIG_PATH)
    p.add_argument("--split",   choices=["test", "train"], default=SPLIT)
    p.add_argument("--output-txt",     default=OUTPUT_TXT)
    p.add_argument("--stats-dir",      default=STATS_DIR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--device",         default=DEVICE)
    return p.parse_args()


def main():
    args   = parse_args()
    device = resolve_device(args.device)
    cfg    = load_config(args.config)

    num_class = cfg["model_args"].get("num_class", 2)

    # Ambil feeder args dari config — gunakan test_feeder_args agar identik dengan training
    feeder_args = cfg.get("test_feeder_args", {}).copy()
    feeder_args["data_path"] = args.skeleton_path
    feeder_args["split"]     = args.split

    model = load_model(cfg, args.weights, device)
    print(f"Device   : {device}")
    print(f"Weights  : {args.weights}")
    print(f"Classes  : {num_class}")
    print(f"Split    : {args.split}")
    print(f"Batch    : {args.batch_size}\n")

    # ── Feeder + DataLoader — SAMA persis dengan pipeline training ────────────
    dataset = Feeder(**feeder_args)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    N = len(dataset)
    print(f"Split '{args.split}': {N} sampel")

    # ── Rekonstruksi nama file asli dari statistics ───────────────────────────
    sample_names = reconstruct_sample_names(args.stats_dir, args.split, N)
    print(f"Nama sampel  : {sample_names[0]} ... {sample_names[-1]}\n")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_probs  = np.zeros((N, num_class), dtype=np.float32)
    all_labels = np.zeros(N, dtype=int)
    offset = 0

    for joint, data, label, _ in loader:
        bs = data.size(0)
        data  = data.float().to(device)
        label = label.long().to(device)
        joint = joint.float()  # topo layer memindahkan ke CPU sendiri

        with torch.no_grad():
            logits, _ = model(data, F.one_hot(label, num_classes=num_class), joint)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        all_probs[offset:offset+bs]  = probs
        all_labels[offset:offset+bs] = label.cpu().numpy()
        offset += bs

        if offset % 200 == 0 or offset == N:
            print(f"  {offset}/{N}...")

    preds      = np.argmax(all_probs, axis=1)
    probs_fall = all_probs[:, FALL_LABEL]

    # ── build rows ────────────────────────────────────────────────────────────
    rows = []
    for i in range(N):
        truth  = LABEL_STR[int(all_labels[i])]
        pred   = LABEL_STR[int(preds[i])]
        status = "BENAR" if preds[i] == all_labels[i] else "SALAH"
        rows.append((sample_names[i], truth, pred, status, float(probs_fall[i])))

    # ── write report ──────────────────────────────────────────────────────────
    acc, prec, rec, f1, tn, fp, fn, tp = write_report(
        args.output_txt, rows, all_labels, preds
    )

    print(f"\n{'─'*50}")
    print(f"  Accuracy   : {acc:.2f}%")
    print(f"  Precision  : {prec:.2f}%")
    print(f"  Recall     : {rec:.2f}%")
    print(f"  F1-Score   : {f1:.2f}%")
    print(f"{'─'*50}")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"{'─'*50}")
    print(f"\nReport saved → {args.output_txt}")


if __name__ == "__main__":
    main()
