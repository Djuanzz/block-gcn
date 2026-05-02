"""
predict_single.py  –  BlockGCN fall detection: single .skeleton file inference.

Cara pakai tercepat  →  edit bagian DEFAULT CONFIG di bawah, lalu:
  python predict_single.py

Atau override lewat argumen:
  python predict_single.py --skeleton-path X --weights Y
"""

import os
import sys
import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from feeders import tools

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT CONFIG  —  edit bagian ini sesuai kebutuhan
# ═══════════════════════════════════════════════════════════════════════════════

# Path ke file .skeleton yang ingin diprediksi
SKELETON_PATH = "data/nturgbd_raw/nturgb+d_skeletons/S001C001P001R001A008.skeleton"

# Path ke file bobot model (.pt)
WEIGHTS_PATH  = "weights/1/runs-71-29678.pt"

# Path ke file konfigurasi YAML
CONFIG_PATH   = "config/nturgbd-cross-subject/default.yaml"

# Device: "auto" (GPU jika tersedia), "cuda", atau "cpu"
DEVICE        = "auto"

# ═══════════════════════════════════════════════════════════════════════════════

# ─── labels ─────────────────────────────────────────────────────────────────────

LABEL_STR = {0: "non-fall", 1: "fall"}


# ─── helpers ─────────────────────────────────────────────────────────────────────

def import_class(name: str):
    import traceback
    mod_str, _, cls_str = name.rpartition(".")
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], cls_str)
    except AttributeError:
        raise ImportError("Class %s cannot be found (%s)" % (cls_str, traceback.format_exception(*sys.exc_info())))


def resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


# ─── skeleton file parser ────────────────────────────────────────────────────────

def parse_skeleton(filepath: str) -> np.ndarray:
    """
    Parse a single NTU RGB+D .skeleton file.

    Returns
    -------
    data : np.ndarray, shape (T_valid, 150)
        150 = 2 bodies × 25 joints × 3 coords (x,y,z).
        Body 2 is all zeros if only one body is present.
        Frames with no body data are dropped.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    idx       = 0
    num_frames = int(lines[idx].strip()); idx += 1

    # Collect per-frame data: dict body_id → list of (frame_idx, joints 25×3)
    body_frames: dict = {}   # body_id(str) → list of (t, np.array 25×3)
    valid_t = 0

    for _ in range(num_frames):
        num_bodies = int(lines[idx].strip()); idx += 1

        if num_bodies == 0:
            continue

        for b in range(num_bodies):
            body_id = lines[idx].strip().split()[0]; idx += 1
            num_joints = int(lines[idx].strip()); idx += 1

            joints = np.zeros((num_joints, 3), dtype=np.float32)
            for j in range(num_joints):
                vals   = lines[idx].strip().split()
                joints[j] = [float(vals[0]), float(vals[1]), float(vals[2])]
                idx += 1

            if body_id not in body_frames:
                body_frames[body_id] = []
            body_frames[body_id].append((valid_t, joints))

        valid_t += 1

    if valid_t == 0:
        raise ValueError(f"No valid frames found in {filepath}")

    # ── select up to 2 bodies ────────────────────────────────────────────────
    # Sort by number of frames (most frames = primary body)
    sorted_bodies = sorted(body_frames.items(),
                           key=lambda kv: len(kv[1]), reverse=True)
    selected = sorted_bodies[:2]  # up to 2 bodies

    # ── build (valid_t, 75) arrays for each body ─────────────────────────────
    body_arrays = []
    for _, frames in selected:
        arr = np.zeros((valid_t, 75), dtype=np.float32)  # 25 joints * 3
        for (t, joints) in frames:
            arr[t] = joints.flatten()
        body_arrays.append(arr)

    # pad to 2 bodies
    while len(body_arrays) < 2:
        body_arrays.append(np.zeros((valid_t, 75), dtype=np.float32))

    data_t150 = np.hstack(body_arrays)   # (valid_t, 150)
    return data_t150


# ─── NTU-style normalization ─────────────────────────────────────────────────────

def apply_seq_translation(data_t150: np.ndarray) -> np.ndarray:
    """
    Subtract the SpineMid (joint 2, 1-based = index 1, 0-based) of body 1
    at the first valid frame from all joints in all frames.

    Mirrors seq_translation() in data/ntu/seq_transformation.py.
    """
    T = data_t150.shape[0]
    data = data_t150.copy()

    # Find first valid frame of body 1 (non-zero)
    first_valid = 0
    for i in range(T):
        if np.any(data[i, :75] != 0):
            first_valid = i
            break

    # SpineMid = joint index 1 (0-based) of body 1 → flat indices 3:6
    origin = data[first_valid, 3:6].copy()   # (3,)

    n_bodies = 2 if not np.all(data[:, 75:] == 0) else 1

    if n_bodies == 2:
        data -= np.tile(origin, (T, 50))   # subtract from all 150 columns
    else:
        data[:, :75] -= np.tile(origin, (T, 25))   # only body-1 columns

    return data


def pad_to_300(data_t150: np.ndarray) -> np.ndarray:
    """Zero-pad (or truncate) to (300, 150) to match NPZ format."""
    T = data_t150.shape[0]
    out = np.zeros((300, 150), dtype=np.float32)
    out[:min(T, 300)] = data_t150[:min(T, 300)]
    return out


def to_ctvm(data_300x150: np.ndarray) -> np.ndarray:
    """(300, 150) → (C=3, T=300, V=25, M=2)"""
    return data_300x150.reshape(300, 2, 25, 3).transpose(3, 0, 2, 1)  # (3, 300, 25, 2)


# ─── feeder-style preprocessing ──────────────────────────────────────────────────

def preprocess(raw_ctvm: np.ndarray, window: int, bone: bool, vel: bool):
    valid = int(np.sum(raw_ctvm.sum(0).sum(-1).sum(-1) != 0))
    data  = tools.valid_crop_resize(raw_ctvm, valid, [1.0], window)
    joint = data.copy()

    if bone:
        from feeders.bone_pairs import ntu_pairs
        bdata = np.zeros_like(data)
        for v1, v2 in ntu_pairs:
            bdata[:, :, v1 - 1] = data[:, :, v1 - 1] - data[:, :, v2 - 1]
        bdata[:, :, 20] = data[:, :, 20]
        data = bdata
    else:
        traj = data[:, :, 20].copy()
        data = data - data[:, :, 20:21]
        data[:, :, 20] = traj

    if vel:
        data[:, :-1] = data[:, 1:] - data[:, :-1]
        data[:, -1]  = 0

    return joint, data


# ─── model ───────────────────────────────────────────────────────────────────────

def load_model(model_class: str, model_args: dict, weights_path: str, device: torch.device):
    import yaml
    Model = import_class(model_class)
    model = Model(**model_args)
    model.to(device)

    raw   = torch.load(weights_path, map_location=device)
    state = OrderedDict((k.replace("module.", ""), v) for k, v in raw.items())
    try:
        model.load_state_dict(state)
    except RuntimeError:
        ms = model.state_dict()
        ms.update({k: v for k, v in state.items() if k in ms})
        model.load_state_dict(ms)

    model.eval()
    return model


def infer(model, joint_np, data_np, num_class, device) -> np.ndarray:
    d = torch.from_numpy(data_np).float().unsqueeze(0).to(device)
    j = torch.from_numpy(joint_np).float().unsqueeze(0).to(device)
    y = torch.zeros(1, num_class, device=device)
    with torch.no_grad():
        logits, _ = model(d, y, j)
        return F.softmax(logits, dim=1).squeeze(0).cpu().numpy()


# ─── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BlockGCN: single .skeleton file inference")
    p.add_argument("--skeleton-path", default=SKELETON_PATH)
    p.add_argument("--weights",       default=WEIGHTS_PATH)
    p.add_argument("--config",        default=CONFIG_PATH)
    p.add_argument("--bone",   action="store_true")
    p.add_argument("--vel",    action="store_true")
    p.add_argument("--device",        default=DEVICE)
    p.add_argument("--window-size", type=int, default=64)
    return p.parse_args()


def main():
    import yaml
    args   = parse_args()
    device = resolve_device(args.device)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    num_class   = cfg["model_args"].get("num_class", 2)
    window_size = args.window_size

    if not args.bone and not args.vel:
        tfa       = cfg.get("test_feeder_args", {})
        args.bone = tfa.get("bone", False)
        args.vel  = tfa.get("vel",  False)

    # ── load model ────────────────────────────────────────────────────────────
    model = load_model(cfg["model"], cfg["model_args"], args.weights, device)

    # ── parse & preprocess skeleton ───────────────────────────────────────────
    skeleton_name = os.path.splitext(os.path.basename(args.skeleton_path))[0]
    print(f"Skeleton : {skeleton_name}")

    data_t150 = parse_skeleton(args.skeleton_path)          # (T_valid, 150)
    data_t150 = apply_seq_translation(data_t150)            # origin centering
    data_300  = pad_to_300(data_t150)                       # (300, 150)
    raw_ctvm  = to_ctvm(data_300)                           # (C, T, V, M)

    joint_np, data_np = preprocess(raw_ctvm, window_size, args.bone, args.vel)

    # ── inference ─────────────────────────────────────────────────────────────
    probs = infer(model, joint_np, data_np, num_class, device)

    pred          = int(np.argmax(probs))
    action        = LABEL_STR[pred]
    fall_pct      = probs[1] * 100
    nonfall_pct   = probs[0] * 100

    print(f"\naction   : {action}")
    print(f"fall     : {fall_pct:.2f}%")
    print(f"non-fall : {nonfall_pct:.2f}%")


if __name__ == "__main__":
    main()
