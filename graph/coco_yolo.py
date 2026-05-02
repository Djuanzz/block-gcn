import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

# COCO 17 keypoints (YOLO11-pose order):
# 0: nose
# 1: left_eye    2: right_eye
# 3: left_ear    4: right_ear
# 5: left_shoulder   6: right_shoulder
# 7: left_elbow      8: right_elbow
# 9: left_wrist      10: right_wrist
# 11: left_hip       12: right_hip
# 13: left_knee      14: right_knee
# 15: left_ankle     16: right_ankle

num_node = 17
self_link = [(i, i) for i in range(num_node)]

# Inward = dari ekstremitas menuju pusat tubuh (area pinggul)
# Format (i, j): i menuju j (j lebih dekat ke pusat)
inward_ori_index = [
    # Kepala → bahu (proxy leher, karena tidak ada joint leher di COCO)
    (1, 0),   (2, 0),   # mata → hidung
    (3, 1),   (4, 2),   # telinga → mata
    (0, 5),   (0, 6),   # hidung → bahu
    # Lengan → bahu
    (9, 7),   (7, 5),   # pergelangan → siku → bahu kiri
    (10, 8),  (8, 6),   # pergelangan → siku → bahu kanan
    # Bahu → pinggul (batang tubuh)
    (5, 11),  (6, 12),
    # Lateral
    (5, 6),             # koneksi antar bahu
    (11, 12),           # koneksi antar pinggul
    # Kaki → pinggul
    (15, 13), (13, 11), # pergelangan kaki → lutut → pinggul kiri
    (16, 14), (14, 12), # pergelangan kaki → lutut → pinggul kanan
]
inward = inward_ori_index   # sudah 0-indexed
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

# ── Grafik kasar level 1: 9 node ───────────────────────────────────────────
# Pemetaan:
#   A1-0 ← nose (0)           — kepala
#   A1-1 ← left_shoulder (5)
#   A1-2 ← right_shoulder (6)
#   A1-3 ← left_elbow (7)     — representasi lengan kiri
#   A1-4 ← right_elbow (8)    — representasi lengan kanan
#   A1-5 ← left_hip (11)
#   A1-6 ← right_hip (12)
#   A1-7 ← left_knee (13)     — representasi kaki kiri
#   A1-8 ← right_knee (14)    — representasi kaki kanan
num_node_1 = 9
indices_1 = [0, 5, 6, 7, 8, 11, 12, 13, 14]  # joint representatif dari grafik penuh
self_link_1 = [(i, i) for i in range(num_node_1)]
inward_ori_index_1 = [
    (0, 1), (0, 2),   # kepala - bahu
    (1, 2),           # lateral bahu
    (3, 1), (4, 2),   # lengan - bahu
    (1, 5), (2, 6),   # bahu - pinggul
    (5, 6),           # lateral pinggul
    (7, 5), (8, 6),   # kaki - pinggul
]
inward_1 = inward_ori_index_1
outward_1 = [(j, i) for (i, j) in inward_1]
neighbor_1 = inward_1 + outward_1

# ── Grafik kasar level 2: 5 node ───────────────────────────────────────────
# Pemetaan:
#   A2-0 ← A1-0 (kepala)
#   A2-1 ← A1-1,3 (bahu kiri + lengan kiri)
#   A2-2 ← A1-2,4 (bahu kanan + lengan kanan)
#   A2-3 ← A1-5,7 (pinggul kiri + kaki kiri)
#   A2-4 ← A1-6,8 (pinggul kanan + kaki kanan)
num_node_2 = 5
indices_2 = [0, 1, 2, 5, 6]  # representatif dari grafik 9-node
self_link_2 = [(i, i) for i in range(num_node_2)]
inward_ori_index_2 = [
    (0, 1), (0, 2),   # kepala - tubuh
    (1, 2),           # lateral tubuh
    (1, 3), (2, 4),   # tubuh - kaki
    (3, 4),           # lateral kaki
]
inward_2 = inward_ori_index_2
outward_2 = [(j, i) for (i, j) in inward_2]
neighbor_2 = inward_2 + outward_2


class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A1 = tools.get_spatial_graph(num_node_1, self_link_1, inward_1, outward_1)
        self.A2 = tools.get_spatial_graph(num_node_2, self_link_2, inward_2, outward_2)
        self.A_binary = tools.edge2mat(neighbor, num_node)
        self.A_norm = tools.normalize_adjacency_matrix(self.A_binary + 2 * np.eye(num_node))
        self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)

        self.A_A1 = (
            (self.A_binary + np.eye(num_node)) /
            np.sum(self.A_binary + np.eye(self.A_binary.shape[0]), axis=1, keepdims=True)
        )[indices_1]

        self.A1_A2 = tools.edge2mat(neighbor_1, num_node_1) + np.eye(num_node_1)
        self.A1_A2 = (self.A1_A2 / np.sum(self.A1_A2, axis=1, keepdims=True))[indices_2]

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
