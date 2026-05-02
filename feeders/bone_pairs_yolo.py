# COCO 17-joint parent-child pairs untuk bone modality (0-indexed)
# Format: (child, parent)  →  bone_vector = pos[child] - pos[parent]
# Root anchor: left_hip (11) — analog dengan spine joint (20) pada NTU
#
# Struktur pohon (dari pusat ke ekstremitas):
#   left_hip(11) ─┬─ right_hip(12) ─┬─ right_shoulder(6) ─┬─ right_elbow(8) ─ right_wrist(10)
#                 │                  │                      └─ ...
#                 │                  └─ right_knee(14) ─ right_ankle(16)
#                 ├─ left_shoulder(5) ─┬─ left_elbow(7) ─ left_wrist(9)
#                 │                    └─ nose(0) ─┬─ left_eye(1) ─ left_ear(3)
#                 │                               └─ right_eye(2) ─ right_ear(4)
#                 └─ left_knee(13) ─ left_ankle(15)

coco_pairs = (
    # Kepala (dari hidung ke bahu, proxy leher)
    (0, 5),    # nose ← left_shoulder
    (1, 0),    # left_eye ← nose
    (2, 0),    # right_eye ← nose
    (3, 1),    # left_ear ← left_eye
    (4, 2),    # right_ear ← right_eye
    # Batang tubuh
    (5, 11),   # left_shoulder ← left_hip
    (6, 12),   # right_shoulder ← right_hip
    (5, 6),    # left_shoulder ← right_shoulder (lateral)
    (12, 11),  # right_hip ← left_hip  [anchor = left_hip]
    # Lengan kiri
    (7, 5),    # left_elbow ← left_shoulder
    (9, 7),    # left_wrist ← left_elbow
    # Lengan kanan
    (8, 6),    # right_elbow ← right_shoulder
    (10, 8),   # right_wrist ← right_elbow
    # Kaki kiri
    (13, 11),  # left_knee ← left_hip
    (15, 13),  # left_ankle ← left_knee
    # Kaki kanan
    (14, 12),  # right_knee ← right_hip
    (16, 14),  # right_ankle ← right_knee
)
