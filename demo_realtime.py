"""
demo_realtime.py
================
Demo real-time fall detection BlockGCN.

Menggabungkan beberapa sekuens skeleton dari dataset, menampilkan
animasi skeleton, dan memprediksi jatuh/tidak-jatuh secara real-time.

Cara pakai:
  python demo_realtime.py
  python demo_realtime.py --classes 8 9 27 42 43 --fps 20
  python demo_realtime.py --classes 8 43 8 43 9 43 --n-per-class 1
  python demo_realtime.py --save demo.mp4
"""

import os
import sys
import glob
import argparse
import warnings
import yaml
from collections import OrderedDict, deque

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
from feeders import tools

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT CONFIG — edit sesuai kebutuhan
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_NPZ     = "data/ntu/NTU60_CS_binary_fall.npz"
DEFAULT_WEIGHTS = "weights/2/runs-62-25916.pt"                              # dir atau file .pt
DEFAULT_CONFIG  = "config/nturgbd-cross-subject/default.yaml"
DEFAULT_CLASSES = [43]                      # urutan aksi yang ditampilkan (1 per kelas = 5 sekuens)
DEFAULT_SPLIT   = "test"                                   # "train" atau "test"
DEFAULT_FPS     = 30                                       # kecepatan animasi
DEFAULT_WINDOW  = 64                                       # window inferensi (frame)
DEFAULT_INFER_EVERY = 3                                    # inferensi tiap N frame
# ═══════════════════════════════════════════════════════════════════════════════

CLASS_INFO = {
    8:  {"name": "Sitting Down",  "is_fall": False, "color": "#2196F3"},
    9:  {"name": "Standing Up",   "is_fall": False, "color": "#4CAF50"},
    27: {"name": "Jump Up",       "is_fall": False, "color": "#FF9800"},
    42: {"name": "Staggering",    "is_fall": False, "color": "#9C27B0"},
    43: {"name": "Falling Down",  "is_fall": True,  "color": "#F44336"},
}

# Tulang skeleton NTU 25-joint untuk visualisasi (0-indexed)
NTU_BONES = [
    (3, 2), (2, 20), (20, 1), (1, 0),          # kepala → spine
    (20, 4), (4, 5), (5, 6), (6, 7),            # lengan kiri
    (7, 21), (7, 22),                            # jari kiri
    (20, 8), (8, 9), (9, 10), (10, 11),         # lengan kanan
    (11, 23), (11, 24),                          # jari kanan
    (0, 12), (12, 13), (13, 14), (14, 15),      # kaki kiri
    (0, 16), (16, 17), (17, 18), (18, 19),      # kaki kanan
]

# Warna per bagian tubuh
BONE_COLORS = {
    "spine": "#FFFFFF",
    "arm_l": "#64B5F6",
    "arm_r": "#EF9A9A",
    "leg_l": "#A5D6A7",
    "leg_r": "#FFCC80",
}

BONE_COLOR_LIST = (
    [BONE_COLORS["spine"]] * 4 +
    [BONE_COLORS["arm_l"]] * 6 +
    [BONE_COLORS["arm_r"]] * 6 +
    [BONE_COLORS["leg_l"]] * 4 +
    [BONE_COLORS["leg_r"]] * 4
)


# ─── utils ───────────────────────────────────────────────────────────────────

def import_class(name):
    mod_str, _, cls_str = name.rpartition(".")
    __import__(mod_str)
    return getattr(sys.modules[mod_str], cls_str)


def find_weights(path):
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        pts = sorted(glob.glob(os.path.join(path, "*.pt")))
        if pts:
            print(f"  Weights ditemukan: {pts[-1]}")
            return pts[-1]
    raise FileNotFoundError(f"Weights tidak ditemukan: {path}")


def load_model(cfg, weights_path, device):
    Model = import_class(cfg["model"])
    model = Model(**cfg["model_args"])
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


# ─── data ────────────────────────────────────────────────────────────────────

def load_samples_by_class(npz_path, split, target_classes, n_per_class, seed=42):
    """
    Muat sampel dari NPZ, pilih n_per_class sampel per kelas.
    Returns: list of dict {class, name, frames_ctvm, valid_frames}
    """
    rng = np.random.default_rng(seed)

    # Rekonstruksi nama dari statistics
    stats_dir = os.path.join(os.path.dirname(npz_path), "statistics")
    names_avail = None
    if os.path.isdir(stats_dir):
        try:
            skes_names = np.loadtxt(os.path.join(stats_dir, "skes_available_name.txt"), dtype=str)
            performer  = np.loadtxt(os.path.join(stats_dir, "performer.txt"), dtype=int)
            label_1b   = np.loadtxt(os.path.join(stats_dir, "label.txt"),     dtype=int)

            TRAIN_IDS = [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]
            TEST_IDS  = [3,6,7,10,11,12,20,21,22,23,24,26,29,30,32,33,36,37,39,40]
            SELECTED  = {8, 9, 27, 42, 43}
            sub_ids   = TRAIN_IDS if split == "train" else TEST_IDS

            indices = []
            for pid in sub_ids:
                indices.extend(np.where(performer == pid)[0].tolist())
            indices  = np.array(indices, dtype=int)
            labs_1b  = label_1b[indices]
            mask     = np.isin(labs_1b, list(SELECTED))
            filt_idx = indices[mask]
            filt_lab = labs_1b[mask]
            names_avail = (skes_names[filt_idx], filt_lab)
        except Exception:
            pass

    npz       = np.load(npz_path)
    x_raw     = npz[f"x_{split}"]          # (N, T, 150)
    y_raw     = npz[f"y_{split}"]          # (N, 2)
    labels    = np.argmax(y_raw, axis=1)   # 0=not_fall, 1=fall
    N, T, _   = x_raw.shape

    # Map binary label → original NTU class (approximate, ambil dari stats)
    # Kita perlu tahu kelas asli tiap sampel
    if names_avail is not None:
        orig_names, orig_labs = names_avail
        assert len(orig_names) == N, "Jumlah sampel NPZ vs stats tidak cocok"
    else:
        orig_labs = np.where(labels == 1, 43, 8)  # fallback kasar

    # Reshape ke (N, C, T, V, M)
    x_ctvm = x_raw.reshape(N, T, 2, 25, 3).transpose(0, 4, 1, 3, 2)

    result = []
    for cls in target_classes:
        if cls not in CLASS_INFO:
            print(f"  [warning] Kelas {cls} tidak dikenal, dilewati")
            continue
        idx_cls = np.where(orig_labs == cls)[0]
        if len(idx_cls) == 0:
            print(f"  [warning] Kelas {cls} tidak ada di split '{split}'")
            continue
        chosen = rng.choice(idx_cls, size=min(n_per_class, len(idx_cls)), replace=False)
        for i in chosen:
            seq = x_ctvm[i]  # (C, T, V, M)
            vf  = int(np.sum(seq.sum(0).sum(-1).sum(-1) != 0))
            name = str(orig_names[i]) if names_avail else f"sample_{i:04d}"
            result.append({
                "class":    cls,
                "name":     name,
                "seq_ctvm": seq[:, :vf],   # (C, T_valid, V, M)
                "valid_frames": vf,
            })
    return result


def chain_sequences(samples):
    """
    Gabungkan list sampel menjadi satu sekuens panjang.
    Returns:
      all_frames : list of (C, V, M) arrays, satu per frame
      frame_meta : list of {class, name, seq_idx, frame_in_seq}
    """
    all_frames = []
    frame_meta = []
    for si, s in enumerate(samples):
        seq = s["seq_ctvm"]           # (C, T_valid, V, M)
        T   = seq.shape[1]
        for t in range(T):
            all_frames.append(seq[:, t])   # (C, V, M)
            frame_meta.append({
                "class":        s["class"],
                "name":         s["name"],
                "seq_idx":      si,
                "frame_in_seq": t,
                "total_frames": T,
            })
    return all_frames, frame_meta


# ─── inferensi ───────────────────────────────────────────────────────────────

def preprocess_buffer(buffer, window_size):
    """
    buffer: deque of (C, V, M) arrays
    Returns: (data_tensor, joint_tensor) siap dimasukkan ke model
    """
    k = len(buffer)
    if k == 0:
        return None, None

    # Stack → (C, k, V, M)
    frames   = np.stack(list(buffer), axis=1)     # (C, k, V, M)
    C, _, V, M = frames.shape

    # Resize ke window_size menggunakan valid_crop_resize
    data  = tools.valid_crop_resize(frames, k, [1.0], window_size)
    joint = data.copy()

    # Trajectory subtraction (joint 20 = thorax/dada)
    traj  = data[:, :, 20].copy()
    data  = data - data[:, :, 20:21]
    data[:, :, 20] = traj

    data_t  = torch.from_numpy(data).float().unsqueeze(0)
    joint_t = torch.from_numpy(joint).float().unsqueeze(0)
    return data_t, joint_t


@torch.no_grad()
def run_infer(model, data_t, joint_t, device, num_class=2):
    data_t = data_t.to(device)
    y      = torch.zeros(1, num_class, device=device)
    logits, _ = model(data_t, y, joint_t)   # joint_t tetap di CPU (Topo.forward handles it)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs  # [p_not_fall, p_fall]


# ─── visualisasi ─────────────────────────────────────────────────────────────

def get_skeleton_xy(frame_ctvm):
    """
    frame_ctvm: (C, V, M) → person0 joints (V, 2) yaitu (x, y) dinormalisasi ke spine_base.
    """
    xyz_p0 = frame_ctvm[:, :, 0]    # (3, 25) — person 0
    xyz_p1 = frame_ctvm[:, :, 1]    # (3, 25) — person 1 (mungkin nol)

    # Pusatkan di spine base (joint 0)
    center = xyz_p0[:, 0:1]          # (3, 1)
    xy0    = (xyz_p0 - center)[[0, 1]].T   # (25, 2): x, y
    xy1    = (xyz_p1 - center)[[0, 1]].T

    has_p1 = float(np.abs(xyz_p1).sum()) > 1e-6
    return xy0, xy1, has_p1


def setup_figure():
    fig = plt.figure(figsize=(14, 7), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(
        2, 2,
        width_ratios=[2.5, 1],
        height_ratios=[10, 1],
        hspace=0.08, wspace=0.25,
    )

    ax_skel  = fig.add_subplot(gs[0, 0], facecolor="#16213e")
    ax_info  = fig.add_subplot(gs[0, 1], facecolor="#16213e")
    ax_prog  = fig.add_subplot(gs[1, :],  facecolor="#16213e")

    # Skeleton axes
    ax_skel.set_xlim(-1.5, 1.5)
    ax_skel.set_ylim(-2.0, 2.0)
    ax_skel.set_aspect("equal")
    ax_skel.axis("off")
    ax_skel.set_title("Skeleton Animasi", color="white", fontsize=11, pad=6)

    # Info axes
    ax_info.axis("off")
    ax_info.set_title("Prediksi Real-time", color="white", fontsize=11, pad=6)

    # Progress axes
    ax_prog.set_xlim(0, 1)
    ax_prog.set_ylim(0, 1)
    ax_prog.axis("off")

    return fig, ax_skel, ax_info, ax_prog


class SkeletonDemo:
    def __init__(self, model, all_frames, frame_meta, args, device):
        self.model       = model
        self.all_frames  = all_frames
        self.frame_meta  = frame_meta
        self.args        = args
        self.device      = device
        self.total_frames = len(all_frames)

        self.buffer     = deque(maxlen=args.window_size)
        self.last_probs = np.array([1.0, 0.0])   # awal: not_fall
        self.infer_cnt  = 0

        self.fig, self.ax_skel, self.ax_info, self.ax_prog = setup_figure()
        self._build_skeleton_artists()
        self._build_info_artists()
        self._build_progress_artist()

    # ── artists ──────────────────────────────────────────────────────────────

    def _build_skeleton_artists(self):
        ax = self.ax_skel
        lw = 2.5
        self.bone_lines = [
            ax.plot([], [], "-", color=BONE_COLOR_LIST[i], lw=lw, alpha=0.85, solid_capstyle="round")[0]
            for i in range(len(NTU_BONES))
        ]
        self.joint_scatter  = ax.scatter([], [], s=30, c="#FFFFFF", zorder=5)
        self.joint_scatter2 = ax.scatter([], [], s=20, c="#AAAAAA", zorder=4, alpha=0.5)

        # Overlay prediction color (rectangle fill)
        self.pred_rect = matplotlib.patches.Rectangle(
            (-1.5, -2.0), 3.0, 4.0, linewidth=3,
            edgecolor="#2196F3", facecolor="none", zorder=10
        )
        ax.add_patch(self.pred_rect)

        # Label prediksi di skeleton
        self.pred_text_skel = ax.text(
            0, 1.85, "", ha="center", va="top",
            fontsize=14, fontweight="bold", color="white", zorder=11
        )

        # Action ground truth
        self.gt_text = ax.text(
            0, -1.9, "", ha="center", va="bottom",
            fontsize=9, color="#AAAAAA", zorder=11
        )

    def _build_info_artists(self):
        ax = self.ax_info
        kw = dict(ha="center", va="center", transform=ax.transAxes)

        self.txt_seq    = ax.text(0.5, 0.93, "", color="#AAAAAA", fontsize=9,  **kw)
        self.txt_name   = ax.text(0.5, 0.86, "", color="#CCCCCC", fontsize=8,  **kw)
        self.txt_gt_lbl = ax.text(0.5, 0.78, "Ground Truth:", color="#888888", fontsize=9, **kw)
        self.txt_gt     = ax.text(0.5, 0.72, "", color="white",  fontsize=11, fontweight="bold", **kw)

        ax.plot([0.05, 0.95], [0.65, 0.65], color="#333366", lw=1, transform=ax.transAxes)

        self.txt_pred_lbl = ax.text(0.5, 0.58, "PREDIKSI MODEL:", color="#888888", fontsize=9, **kw)
        self.txt_pred     = ax.text(0.5, 0.47, "...", color="white",  fontsize=20, fontweight="bold", **kw)
        self.txt_conf     = ax.text(0.5, 0.37, "fall prob: -", color="#AAAAAA", fontsize=9, **kw)

        ax.plot([0.05, 0.95], [0.30, 0.30], color="#333366", lw=1, transform=ax.transAxes)

        # Bar chart probabilitas
        self.bar_bg0 = matplotlib.patches.FancyBboxPatch(
            (0.05, 0.14), 0.90, 0.10,
            boxstyle="round,pad=0.01", transform=ax.transAxes,
            facecolor="#333333", edgecolor="none", zorder=1
        )
        self.bar_fg0 = matplotlib.patches.FancyBboxPatch(
            (0.05, 0.14), 0.00, 0.10,
            boxstyle="round,pad=0.01", transform=ax.transAxes,
            facecolor="#F44336", edgecolor="none", zorder=2
        )
        ax.add_patch(self.bar_bg0)
        ax.add_patch(self.bar_fg0)
        self.txt_bar_lbl = ax.text(0.5, 0.10, "Fall Probability",
                                   color="#666666", fontsize=8, **kw)
        self.txt_bar_l = ax.text(0.05, 0.07, "0%",   color="#555555", fontsize=7,
                                  ha="left", va="top", transform=ax.transAxes)
        self.txt_bar_r = ax.text(0.95, 0.07, "100%", color="#555555", fontsize=7,
                                  ha="right", va="top", transform=ax.transAxes)

        # Buffer fill indicator
        self.txt_buf = ax.text(0.5, 0.02, "", color="#555555", fontsize=7, **kw)

    def _build_progress_artist(self):
        ax = self.ax_prog
        self.prog_bg = matplotlib.patches.Rectangle(
            (0.01, 0.2), 0.98, 0.6,
            transform=ax.transAxes, facecolor="#333333", edgecolor="none"
        )
        self.prog_fg = matplotlib.patches.Rectangle(
            (0.01, 0.2), 0.0, 0.6,
            transform=ax.transAxes, facecolor="#2196F3", edgecolor="none"
        )
        ax.add_patch(self.prog_bg)
        ax.add_patch(self.prog_fg)
        self.txt_prog = ax.text(
            0.5, 0.5, "", ha="center", va="center",
            transform=ax.transAxes, color="white", fontsize=8
        )

        # Sequence separators — drawn once
        seq_changes = []
        for i in range(1, len(self.frame_meta)):
            if self.frame_meta[i]["seq_idx"] != self.frame_meta[i-1]["seq_idx"]:
                seq_changes.append(i / self.total_frames)
        for x in seq_changes:
            ax.plot([x, x], [0.1, 0.9], color="#FFFFFF", lw=0.8,
                    alpha=0.4, transform=ax.transAxes)

    # ── update per frame ─────────────────────────────────────────────────────

    def _update_skeleton(self, frame_ctvm, pred_label, fall_prob):
        xy0, xy1, has_p1 = get_skeleton_xy(frame_ctvm)

        # Gambar tulang person 0
        for li, (j1, j2) in enumerate(NTU_BONES):
            x = [xy0[j1, 0], xy0[j2, 0]]
            y = [xy0[j1, 1], xy0[j2, 1]]
            self.bone_lines[li].set_data(x, y)

        # Joints
        self.joint_scatter.set_offsets(xy0)
        if has_p1:
            self.joint_scatter2.set_offsets(xy1)
        else:
            self.joint_scatter2.set_offsets(np.empty((0, 2)))

        # Warna border berdasarkan prediksi
        color = "#F44336" if pred_label == 1 else "#4CAF50"
        self.pred_rect.set_edgecolor(color)

        # Label prediksi di skeleton
        label_str = "FALL!" if pred_label == 1 else "SAFE"
        self.pred_text_skel.set_text(label_str)
        self.pred_text_skel.set_color(color)

    def _update_info(self, meta, pred_label, fall_prob, buf_len):
        info = CLASS_INFO.get(meta["class"], {})
        gt_str  = info.get("name", f"A{meta['class']:03d}")
        gt_fall = info.get("is_fall", False)
        gt_col  = "#F44336" if gt_fall else "#4CAF50"

        self.txt_seq.set_text(
            f"Sekuens {meta['seq_idx']+1} / {len(set(m['seq_idx'] for m in self.frame_meta))}"
            f"  |  Frame {meta['frame_in_seq']+1}/{meta['total_frames']}"
        )
        self.txt_name.set_text(meta["name"])
        self.txt_gt.set_text(gt_str)
        self.txt_gt.set_color(gt_col)

        # Prediksi
        pred_str = "FALL" if pred_label == 1 else "NOT FALL"
        pred_col = "#F44336" if pred_label == 1 else "#4CAF50"
        self.txt_pred.set_text(pred_str)
        self.txt_pred.set_color(pred_col)
        self.txt_conf.set_text(f"fall prob: {fall_prob:.1%}")

        # Ground truth di skeleton
        self.gt_text.set_text(f"[GT] {gt_str}")

        # Bar probabilitas
        bar_w = 0.90 * fall_prob
        self.bar_fg0.set_width(bar_w)
        bar_col = (
            "#F44336" if fall_prob > 0.6 else
            "#FF9800" if fall_prob > 0.35 else
            "#4CAF50"
        )
        self.bar_fg0.set_facecolor(bar_col)

        # Buffer fill
        fill_pct = buf_len / self.args.window_size * 100
        self.txt_buf.set_text(f"buffer: {buf_len}/{self.args.window_size} ({fill_pct:.0f}%)")

    def _update_progress(self, frame_idx):
        prog = (frame_idx + 1) / self.total_frames
        self.prog_fg.set_width(0.98 * prog)

        meta = self.frame_meta[frame_idx]
        info = CLASS_INFO.get(meta["class"], {})
        # Update warna progress berdasarkan aksi saat ini
        col = info.get("color", "#2196F3")
        self.prog_fg.set_facecolor(col)

        self.txt_prog.set_text(
            f"Frame global: {frame_idx+1}/{self.total_frames}  |  "
            f"Aksi: {info.get('name', '?')}"
        )

    # ── animasi ──────────────────────────────────────────────────────────────

    def animate_frame(self, frame_idx):
        frame_ctvm = self.all_frames[frame_idx]   # (C, V, M)
        meta       = self.frame_meta[frame_idx]

        # Tambahkan ke buffer
        self.buffer.append(frame_ctvm)
        self.infer_cnt += 1

        # Jalankan inferensi tiap N frame
        if self.infer_cnt >= self.args.infer_every or frame_idx == 0:
            self.infer_cnt = 0
            data_t, joint_t = preprocess_buffer(self.buffer, self.args.window_size)
            if data_t is not None:
                self.last_probs = run_infer(
                    self.model, data_t, joint_t, self.device, num_class=2
                )

        pred_label = int(np.argmax(self.last_probs))
        fall_prob  = float(self.last_probs[1])

        self._update_skeleton(frame_ctvm, pred_label, fall_prob)
        self._update_info(meta, pred_label, fall_prob, len(self.buffer))
        self._update_progress(frame_idx)

        return (
            *self.bone_lines,
            self.joint_scatter, self.joint_scatter2,
            self.pred_rect, self.pred_text_skel, self.gt_text,
            self.txt_seq, self.txt_name, self.txt_gt,
            self.txt_pred, self.txt_conf,
            self.bar_fg0, self.txt_buf,
            self.prog_fg, self.txt_prog,
        )

    def run(self, save_path=None):
        interval_ms = int(1000 / self.args.fps)
        ani = animation.FuncAnimation(
            self.fig,
            self.animate_frame,
            frames=self.total_frames,
            interval=interval_ms,
            blit=False,
            repeat=False,
        )

        if save_path:
            print(f"Menyimpan video ke: {save_path}  (bisa memakan waktu beberapa menit) ...")
            writer = animation.FFMpegWriter(fps=self.args.fps, bitrate=1800)
            ani.save(save_path, writer=writer, dpi=120)
            print("Selesai.")
        else:
            plt.show()


# ─── main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BlockGCN Real-time Fall Detection Demo")
    p.add_argument("--npz",         default=DEFAULT_NPZ)
    p.add_argument("--weights",     default=DEFAULT_WEIGHTS)
    p.add_argument("--config",      default=DEFAULT_CONFIG)
    p.add_argument("--split",       default=DEFAULT_SPLIT, choices=["train", "test"])
    p.add_argument("--classes",     nargs="+", type=int, default=DEFAULT_CLASSES,
                   help="Urutan kelas yang ditampilkan, mis: --classes 8 9 43 8 43")
    p.add_argument("--n-per-class", type=int, default=10,
                   help="Jumlah sampel diambil per kelas (default 3 = 15 sekuens total)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--fps",         type=int, default=DEFAULT_FPS)
    p.add_argument("--window-size", type=int, default=DEFAULT_WINDOW, dest="window_size")
    p.add_argument("--infer-every", type=int, default=DEFAULT_INFER_EVERY, dest="infer_every")
    p.add_argument("--device",      default="auto")
    p.add_argument("--save",        default=None, metavar="FILE.mp4",
                   help="Simpan animasi ke file MP4 (butuh ffmpeg)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{'='*55}")
    print("  BlockGCN — Demo Real-time Fall Detection")
    print(f"{'='*55}")

    # Load config & model
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    weights_path = find_weights(args.weights)
    model = load_model(cfg, weights_path, device)
    print(f"  Device  : {device}")
    print(f"  Weights : {weights_path}")
    print(f"  Kelas   : {[CLASS_INFO[c]['name'] for c in args.classes if c in CLASS_INFO]}")
    print(f"  FPS     : {args.fps}")

    # Load & chain sequences
    print(f"\n  Memuat sampel dari NPZ ({args.split} split) ...")
    samples = load_samples_by_class(
        args.npz, args.split, args.classes, args.n_per_class, args.seed
    )

    if not samples:
        sys.exit("  [ERROR] Tidak ada sampel yang berhasil dimuat.")

    print(f"\n  Sekuens yang akan ditampilkan ({len(samples)} total):")
    total_f = 0
    for s in samples:
        print(f"    [{s['class']:2d}] {CLASS_INFO[s['class']]['name']:<15} | "
              f"{s['name']}  ({s['valid_frames']} frame)")
        total_f += s["valid_frames"]
    print(f"  Total frame gabungan : {total_f}  (~{total_f/30:.0f} detik @ 30fps asli)")

    all_frames, frame_meta = chain_sequences(samples)
    print(f"\n  Menampilkan animasi ...")
    print(f"{'='*55}\n")

    demo = SkeletonDemo(model, all_frames, frame_meta, args, device)
    demo.run(save_path=args.save)


if __name__ == "__main__":
    main()
