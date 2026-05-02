import numpy as np
from torch.utils.data import Dataset
from feeders import tools


class Feeder(Dataset):
    """
    Data feeder untuk skeleton YOLO11-pose dari NTU RGB+D.

    Format NPZ yang diharapkan (output dari prepare_dataset_yolo.py):
        x_train / x_test : (N, T=300, 51)
            51 = 1 orang × 17 joint × 3 (x_norm, y_norm, confidence)
        y_train / y_test : (N, 2) one-hot  [non-fall, fall]

    Shape internal setelah load_data:
        self.data : (N, C=3, T=300, V=17, M=1)
    """

    def __init__(self, data_path, label_path=None, p_interval=1, split='train',
                 random_choose=False, random_shift=False, random_move=False,
                 random_rot=False, window_size=-1, normalization=False,
                 debug=False, use_mmap=False, bone=False, vel=False):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split hanya mendukung train/test')

        if self.debug:
            self.data = self.data[:100]
            self.label = self.label[:100]
            self.sample_name = self.sample_name[:100]

        # (N, T, 51) → reshape ke (N, T, 1, 17, 3) → transpose ke (N, C=3, T, V=17, M=1)
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 1, 17, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = (
            data.mean(axis=2, keepdims=True)
                .mean(axis=4, keepdims=True)
                .mean(axis=0)
        )
        self.std_map = (
            data.transpose((0, 2, 4, 1, 3))
                .reshape((N * T * M, C * V))
                .std(axis=0)
                .reshape((C, 1, V, 1))
        )

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]        # (C=3, T=300, V=17, M=1)
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        # Hitung frame valid (minimal satu joint non-zero)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        # Crop + resize ke window_size (identik dengan feeder_ntu.py)
        data_numpy = tools.valid_crop_resize(
            data_numpy, valid_frame_num, self.p_interval, self.window_size
        )

        if self.random_move and valid_frame_num > 0:
            data_numpy = tools.random_move(data_numpy)

        joint = data_numpy.copy()

        # random_rot: hanya aktifkan jika paham dampaknya pada data 2D
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
            joint = data_numpy

        if self.bone:
            from .bone_pairs_yolo import coco_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in coco_pairs:
                bone_data_numpy[:, :, v1] = data_numpy[:, :, v1] - data_numpy[:, :, v2]
            # Simpan trajektori left_hip (11) sebagai referensi
            # — analog dengan joint spine (20) pada feeder_ntu.py
            bone_data_numpy[:, :, 11] = data_numpy[:, :, 11]
            data_numpy = bone_data_numpy
        else:
            # Pusatkan setiap frame pada titik tengah pinggul
            # hip_center shape: (C, T, 1, M)
            hip_center = (
                data_numpy[:, :, 11:12, :] + data_numpy[:, :, 12:13, :]
            ) / 2.0
            trajectory = hip_center.copy()
            data_numpy = data_numpy - hip_center
            # Kembalikan informasi trajektori pada left_hip
            data_numpy[:, :, 11:12, :] = trajectory

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return joint, data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
