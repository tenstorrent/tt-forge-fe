# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch

import forge
import forge.op
from forge import DepricatedVerifyConfig, ForgeModule
from forge.verify.backend import verify_module
from forge.verify.config import TestKind


class PtFuseTMMultiUser(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "segformer.encoder.layer_norm.0.weight",
            forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "segformer.encoder.layer_norm.0.bias",
            forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "segformer.encoder.patch_embeddings.1.proj.weight",
            forge.Parameter(*(64, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "segformer.encoder.patch_embeddings.1.proj.bias",
            forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, input):
        layernorm_340 = forge.op.Layernorm(
            "",
            input,
            self.get_parameter("segformer.encoder.layer_norm.0.weight"),
            self.get_parameter("segformer.encoder.layer_norm.0.bias"),
            dim=-1,
            epsilon=1e-05,
        )
        reshape_341 = forge.op.Reshape("", layernorm_340, shape=(1, 128, 128, 32))
        transpose_342 = forge.op.Transpose("", reshape_341, dim0=-3, dim1=-1)
        transpose_343 = forge.op.Transpose("", transpose_342, dim0=-2, dim1=-1)
        conv2d_344 = forge.op.Conv2d(
            "",
            transpose_343,
            self.get_parameter("segformer.encoder.patch_embeddings.1.proj.weight"),
            self.get_parameter("segformer.encoder.patch_embeddings.1.proj.bias"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        reshape_783 = forge.op.Reshape("", transpose_343, shape=(1, 32, 16384))
        transpose_784 = forge.op.Transpose("", reshape_783, dim0=-2, dim1=-1)
        reshape_785 = forge.op.Reshape("", transpose_784, shape=(16384, 32))
        return conv2d_344, reshape_785


def test_fuse_tm_sequence_multi_user(test_device):
    """
       Test case to fuse tm sequence when there are multiple user for the matched pattern in optimization forge passes

       Pattern to Match:
                vslice
                transpose(-3, -1)
                transpose(-2, -1)
                reshape


       Pattern to Replace:
                transpose(-2, -1)


       Graph before fusing:

                  Input
            [1, 1, 16384, 32]
                    |
                    |
                vslice_1(128,)
            [1, 128, 128, 32]
                    |
                    |
                transpose_1(-3, -1, 32)
            [1, 32, 128, 128]
                    |
                    |
                transpose_2(-2, -1, -1)
            [1, 32, 128, 128]
                    /\
                   /  \
                  /    \
           reshape_1   reshape_2
    [1, 1, 32, 16384]   [1, 32, 16384]


    Graph after fusing:

                  Input
              [1, 1, 16384, 32]
                    |
                    |
            fused_op_transpose_1(-2, -1, -1)
              [1, 1, 32, 16384]
                     \
                      \
                      reshape_2
                      [1, 32, 16384]


    If there are multiple user at the last pattern matched node which are same op and same shape
    (i.e reshape_1(1, 1, 32, 16384) and reshape_2(1, 32, 16384)), in that cases reshape_1 will be fused
    and reshape_2 will be connected to the fused_op_transpose_1.

    """

    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = forge.DataFormat.Float16_b
    os.environ["FORGE_RIBBON2"] = "1"

    tt_model = PtFuseTMMultiUser("fuse_tm_sequence_multi_user")

    pt_tensor = forge.Tensor.create_from_torch(torch.rand((1, 16384, 32)))

    verify_module(
        tt_model,
        input_shapes=(pt_tensor.shape,),
        inputs=[(pt_tensor,)],
        verify_cfg=DepricatedVerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )
