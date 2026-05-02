"""
model/BlockGCN.py  —  BlockGCN untuk binary fall detection
============================================================
Arsitektur mengikuti paper BlockGCN:
  - 10 blok TCN-GCN dengan multi-scale temporal convolution
  - Relative positional encoding via k-hop distance dalam GCN
  - Topological features (persistent homology) di-fuse ke setiap blok

Versi ini (bersih):
  - forward(x)  →  hanya 1 argumen (konsisten dengan feeder 2-return-value)
  - Return: logits (N, num_class)  saja
  - nn.Identity() / None untuk residual (bukan lambda)
  - Topo: device-transparent (tidak paksa ke CPU)
  - TopoTrans: guard untuk single-sample inference (batch_size=1)
"""

import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torch_topological.nn.data import make_tensor
from torch_topological.nn import VietorisRipsComplex
from torch_topological.nn.layers import StructureElementLayer


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


# ── Temporal Convolution ──────────────────────────────────────────────────────

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
        )
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        return self.bn(self.conv(x))


class MultiScale_TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilations=[1, 2, 3, 4], residual=False, residual_kernel_size=1):
        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, \
            '# out channels should be multiples of # branches'

        self.num_branches = len(dilations) + 2
        branch_channels   = out_channels // self.num_branches

        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(branch_channels, branch_channels,
                             kernel_size=ks, stride=stride, dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels),
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels,
                      kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels),
        ))

        if not residual:
            self.residual = None
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = TemporalConv(in_channels, out_channels,
                                         kernel_size=residual_kernel_size, stride=stride)

        self.apply(weights_init)

    def forward(self, x):
        res          = self.residual(x) if self.residual is not None else 0
        branch_outs  = [b(x) for b in self.branches]
        return torch.cat(branch_outs, dim=1) + res


# ── Graph Convolution ─────────────────────────────────────────────────────────

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True, alpha=False):
        super().__init__()
        self.out_c     = out_channels
        self.in_c      = in_channels
        self.num_heads = 8 if in_channels > 8 else 1

        self.fc1 = nn.Parameter(
            torch.stack([
                torch.stack([torch.eye(A.shape[-1])
                             for _ in range(self.num_heads)], dim=0)
                for _ in range(3)
            ], dim=0),
            requires_grad=True,
        )
        self.fc2 = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, groups=self.num_heads)
            for _ in range(3)
        ])

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = nn.Identity()

        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        # k-hop relative position encoding
        h1 = A.sum(0)
        h1[h1 != 0] = 1

        V  = A.shape[-1]
        h  = [None] * V
        h[0] = np.eye(V)
        h[1] = h1
        self.hops = 0 * h[0]
        for i in range(2, V):
            h[i] = h[i - 1] @ h1.T
            h[i][h[i] != 0] = 1

        for i in range(V - 1, 0, -1):
            if np.any(h[i] - h[i - 1]):
                h[i]       = h[i] - h[i - 1]
                self.hops += i * h[i]

        self.hops = torch.tensor(self.hops).long()
        self.rpe  = nn.Parameter(
            torch.zeros((3, self.num_heads, self.hops.max() + 1)))

        if alpha:
            self.alpha = nn.Parameter(torch.ones(1, self.num_heads, 1, 1, 1))
        else:
            self.alpha = 1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def L2_norm(self, weight):
        return torch.norm(weight, 2, dim=-2, keepdim=True) + 1e-4

    def forward(self, x):
        N, C, T, V = x.size()
        y       = None
        pos_emb = self.rpe[:, :, self.hops]

        for i in range(3):
            w1 = self.fc1[i] / self.L2_norm(self.fc1[i])
            w1 = w1 + pos_emb[i] / self.L2_norm(pos_emb[i])
            x_in = x.view(N, self.num_heads, C // self.num_heads, T, V)
            z    = torch.einsum("nhctv,hvw->nhctw",
                                (x_in, w1)).contiguous().view(N, -1, T, V)
            z    = self.fc2[i](z)
            y    = z + y if y is not None else z

        y  = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True,
                 adaptive=True, kernel_size=5, dilations=[1, 2],
                 num_point=17, num_heads=16, alpha=False):
        super().__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A,
                              adaptive=adaptive, alpha=alpha)
        self.tcn1 = MultiScale_TemporalConv(
            out_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            dilations=dilations, residual=False,
        )
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = None
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = unit_tcn(in_channels, out_channels,
                                     kernel_size=1, stride=stride)

    def forward(self, x):
        res = self.residual(x) if self.residual is not None else 0
        return self.relu(self.tcn1(self.gcn1(x)) + res)


# ── Topological Encoding ──────────────────────────────────────────────────────

class TopoTrans(nn.Module):
    """Transform fitur topologi (N, 64) ke dimensi GCN (N, out_dim, 1, 1)."""
    def __init__(self, out_dim):
        super().__init__()
        self.mlp = nn.Linear(64, out_dim)
        self.bn  = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Guard: pastikan selalu 2D meski N=1 saat inferensi single sample
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.relu(self.bn(self.mlp(x)))
        return x.unsqueeze(2).unsqueeze(3)   # (N, out_dim, 1, 1)


class Topo(nn.Module):
    """Persistent homology via Vietoris-Rips untuk fitur topologi skeleton."""
    def __init__(self, dims=0):
        super().__init__()
        self.vr   = VietorisRipsComplex(dim=dims)
        self.pl   = StructureElementLayer(n_elements=64)
        self.relu = nn.ReLU()

    def L2_norm(self, weight):
        return torch.norm(weight, 2, dim=1)

    def forward(self, x):
        # x: (N, M, C, T, V)
        N_batch = x.shape[0]
        device  = x.device

        # VietorisRipsComplex harus jalan di CPU
        x = x.cpu()

        x = x.mean(1)                           # (N, C, T, V)
        x = x.unsqueeze(-1) - x.unsqueeze(-2)   # (N, C, T, V, V)
        x = x.mean(-3)                          # (N, C, V, V)
        x = self.L2_norm(x)                     # (N, V, V)

        x_min = x.min()
        x_max = x.max()
        if x_max - x_min > 1e-8:
            x = (x - x_min) / (x_max - x_min)
        else:
            # Data hampir nol (batch zero-padded) → fallback zeros
            return torch.zeros(N_batch, 64, device=device)

        try:
            x    = self.vr(x)
            x    = make_tensor(x)
            self.pl = self.pl.cpu()
            x    = self.pl(x)
        except (ValueError, RuntimeError):
            # Persistence diagram kosong → fallback zeros
            return torch.zeros(N_batch, 64, device=device)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        return x.to(device)   # (N, 64) dikembalikan ke device asli


# ── Model Utama ───────────────────────────────────────────────────────────────

class Model(nn.Module):
    """
    BlockGCN untuk binary fall detection.

    Args (via config):
      num_class  : jumlah kelas (2 untuk fall/not_fall)
      num_point  : jumlah joint (17 untuk YOLO, 25 untuk NTU)
      num_person : 1 (single person)
      graph      : nama class graph (mis. 'graph.ntu_rgb_d.Graph')
      in_channels: 3 (x,y,conf atau x,y,z)
      drop_out   : dropout sebelum classifier (0.3 direkomendasikan)

    forward(x) → logits (N, num_class)
      x: (N, C=3, T, V=num_point, M=1)
    """

    def __init__(self, num_class=2, num_point=25, num_person=1,
                 graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, num_set=3, alpha=False,
                 window_size=150, **kwargs):
        super().__init__()

        if graph is None:
            raise ValueError("Argumen 'graph' wajib diisi.")

        Graph      = import_class(graph)
        self.graph = Graph(**graph_args)
        A          = self.graph.A   # (3, V, V)

        self.num_class  = num_class
        self.num_point  = num_point
        self.num_person = num_person

        # data_bn dipanggil SETELAH to_joint_embedding → dim = num_person * 128 * num_point
        self.data_bn = nn.BatchNorm1d(num_person * 128 * num_point)

        # Joint embedding: koordinat → 128 dim
        self.to_joint_embedding = nn.Linear(in_channels, 128)
        self.pos_embedding       = nn.Parameter(torch.randn(1, num_point, 128))

        # 10 blok TCN-GCN
        self.l1  = TCN_GCN_unit(128, 128, A, adaptive=adaptive, alpha=alpha)
        self.l2  = TCN_GCN_unit(128, 128, A, adaptive=adaptive, alpha=alpha)
        self.l3  = TCN_GCN_unit(128, 128, A, adaptive=adaptive, alpha=alpha)
        self.l4  = TCN_GCN_unit(128, 128, A, adaptive=adaptive, alpha=alpha)
        self.l5  = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive, alpha=alpha)
        self.l6  = TCN_GCN_unit(256, 256, A, adaptive=adaptive, alpha=alpha)
        self.l7  = TCN_GCN_unit(256, 256, A, adaptive=adaptive, alpha=alpha)
        self.l8  = TCN_GCN_unit(256, 256, A, stride=2, adaptive=adaptive, alpha=alpha)
        self.l9  = TCN_GCN_unit(256, 256, A, adaptive=adaptive, alpha=alpha)
        self.l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, alpha=alpha)

        # Topological encoding — 10 transformer (satu per blok)
        self.topo = Topo()
        self.t0   = TopoTrans(128)
        self.t1   = TopoTrans(128)
        self.t2   = TopoTrans(128)
        self.t3   = TopoTrans(128)
        self.t4   = TopoTrans(128)
        self.t5   = TopoTrans(256)
        self.t6   = TopoTrans(256)
        self.t7   = TopoTrans(256)
        self.t8   = TopoTrans(256)
        self.t9   = TopoTrans(256)

        # Classifier
        self.fc       = nn.Linear(256, num_class)
        self.drop_out = nn.Dropout(p=drop_out) if drop_out > 0 else nn.Identity()

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        """
        Args:
            x: (N, C=3, T, V, M=1)
        Returns:
            logits: (N, num_class)
        """
        N, C, T, V, M = x.size()

        # ── 1. Topological encoding ──────────────────────────────────────────
        joint_topo = rearrange(x, 'n c t v m -> n m c t v').contiguous()
        a = self.topo(joint_topo)   # (N, 64)

        # ── 2. Joint embedding ───────────────────────────────────────────────
        x = rearrange(x, 'n c t v m -> (n m t) v c').contiguous()
        x = self.to_joint_embedding(x)            # (N*M*T, V, 128)
        x = x + self.pos_embedding[:, :V]         # positional encoding

        # BatchNorm pada representasi flattened (N, M*V*128, T)
        x = rearrange(x, '(n m t) v c -> n (m v c) t',
                      n=N, m=M, t=T).contiguous()
        x = self.data_bn(x)

        # Reshape ke (N*M, 128, T, V) untuk GCN
        x = x.view(N, M, V, 128, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, 128, T, V)

        # ── 3. 10 blok GCN + TCN dengan fusi topologi ───────────────────────
        x = self.l1(x  + self.t0(a))
        x = self.l2(x  + self.t1(a))
        x = self.l3(x  + self.t2(a))
        x = self.l4(x  + self.t3(a))
        x = self.l5(x  + self.t4(a))
        x = self.l6(x  + self.t5(a))
        x = self.l7(x  + self.t6(a))
        x = self.l8(x  + self.t7(a))
        x = self.l9(x  + self.t8(a))
        x = self.l10(x + self.t9(a))

        # ── 4. Global average pooling + classifier ───────────────────────────
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1).mean(3).mean(1)   # (N, 256)
        x = self.drop_out(x)
        return self.fc(x)   # (N, num_class)
