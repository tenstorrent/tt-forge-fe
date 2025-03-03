# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from test.mlir.nerf.spherical_harmonics import eval_sh


class NeRFHead(nn.Module):
    def __init__(self, W, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Linear(W, W)
        self.relu1 = nn.ReLU(False)
        self.layer2 = nn.Linear(W, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        return x


class NeRFEncoding(nn.Module):
    def __init__(self, in_dim, W, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Linear(in_dim, W)
        self.relu1 = nn.ReLU(False)
        self.layer2 = nn.Linear(W, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        return x


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, in_channels_xyz=63, in_channels_dir=27, deg=2):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.deg = deg

        for i in range(D):
            if i == 0:
                layer = NeRFEncoding(in_channels_xyz, W, W)
            else:
                layer = NeRFEncoding(W, W, W)
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.sigma = NeRFHead(W, 1)
        self.sh = NeRFHead(W, 32)

    def forward(self, xyz):
        input_xyz = xyz

        xyz_ = input_xyz
        for i in range(self.D):
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        sh = self.sh(xyz_)
        return sigma, sh

    def postprocess(self, sigma, sh, dirs=None):
        sh = sh[:, :27]
        rgb = eval_sh(deg=self.deg, sh=sh.reshape(-1, 3, (self.deg + 1) ** 2), dirs=dirs)
        rgb = torch.sigmoid(rgb)
        return sigma, rgb, sh
