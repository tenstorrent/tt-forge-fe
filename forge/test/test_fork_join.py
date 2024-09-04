# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import List

import pytest
import torch

import forge
from forge.verify import verify_module, VerifyConfig
from forge import DataFormat, ForgeModule

shape = (128, 768)

def get_relaxed_atol_pcc(test_kind, test_device):
    """
    Figure out reasonable pcc/atol for training on silicon
    """
    training_atol = 0.3
    training_pcc = 0.95
    if test_device.is_silicon():
        training_pcc = 0.85
    inference_atol = 0.1
    inference_pcc = 0.95
    relative_atol = training_atol if test_kind.is_training() else inference_atol
    if test_device.is_silicon() and test_kind.is_training():
        relative_atol *= 3.5
    pcc = training_pcc if test_kind.is_training() else inference_pcc

    return relative_atol, pcc

class ForkJoinVariant(forge.ForgeModule):

    def __init__(self, name, input_shape, config):
        super().__init__(name)
        self.weights1 = forge.Parameter(1, input_shape[1], input_shape[1], requires_grad=True) 
        self.input_shape = input_shape
        self.config = config

    def forward(self, act1):

        # fork
        if self.config[0] == "e":
            fork = forge.op.Gelu("gelu_fork", act1)
        elif self.config[0] == "m":
            fork = forge.op.Matmul("matmul_fork", act1, act1)
        else:
            raise TypeError("Unexpected value in configuration of fork-join test")

        # right
        if self.config[1] == "e":
            right = forge.op.Add("add_long_path", fork, self.weights1)
        elif self.config[1] == "m":
            right = forge.op.Matmul("matmul_long_path", fork, self.weights1)
        else:
            raise TypeError("Unexpected value in configuration of fork-join test")

        # join
        if self.config[2] == "e":
            join = forge.op.Add("add_join", fork, right)
        elif self.config[2] == "m":
            join = forge.op.Matmul("matmul_join", fork, right)
        else:
            raise TypeError("Unexpected value in configuration of fork-join test")

        return join

@pytest.mark.parametrize("input_shape", [(128,128), (256,256), (512,512)], ids=["128","256","512"])
@pytest.mark.parametrize("config", ["mem", "mmm", "eme", "emm"], ids=["mem", "mmm", "eme", "emm"])
def test_fork_join_variant(test_kind, test_device, input_shape, config):
    """
        input_shape: input shape of the tensor in the fork-join.
        config: string that tells us type of each op in the simple fork-join. first character describes fork node, second describes op on the longer path and third describes join node.
                if config is "m" then apropriate node is matmul, and if it is "e", then node is element-wise op.
    """
    num_in_channels = 1
    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    verify_module(ForkJoinVariant("test_fork_join_variant", input_shape, config), [(1, num_in_channels, input_shape[0], input_shape[1])],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, pcc=pcc, relative_atol=relative_atol))

class ForkJoin(forge.ForgeModule):

    def __init__(self, name, stack_factor: int = 12):
        super().__init__(name)
        self.weights1 = forge.Parameter(stack_factor, shape[1] // stack_factor, shape[1] // stack_factor, requires_grad=True) 
        self.weights2 = forge.Parameter(1, shape[1], shape[1], requires_grad=True) 
        self.weights3 = forge.Parameter(stack_factor, shape[1] // stack_factor, shape[1] // stack_factor, requires_grad=True) 
        self.stack_factor = stack_factor

    def forward(self, act1):

        # input slice
        sliced = forge.op.HSlice("slice", act1, self.stack_factor)

        # fork, t=stack_factor
        fork = forge.op.Gelu("gelu", sliced)

        # right
        right = forge.op.Matmul("matmul_1", fork, self.weights1)
        right = forge.op.HStack("stack_branch", right)
        right = forge.op.Matmul("matmul_2a_t1", right, self.weights2)
        right = forge.op.Matmul("matmul_2b_t1", right, self.weights2)
        right = forge.op.HSlice("slice_branch", right, self.stack_factor)
        right = forge.op.Matmul("matmul_3", right, self.weights3)

        # join
        join = forge.op.Add("join", fork, right)
        return join

@pytest.mark.parametrize("format", [DataFormat.Bfp8_b, DataFormat.Float16_b], ids=["bfp8", "fp16"])
def test_fork_join(test_kind, test_device, format):
    if test_device.arch == forge.BackendDevice.Blackhole:
         pytest.skip("Skip until ForgeBackend#2628 is consumed.")

    microbatch_count = 16

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    verify_module(ForkJoin("fork_join"), [(microbatch_count, *shape)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, pcc=pcc, relative_atol=relative_atol,
                fp32_fallback=format))

class ForkJoinWithBuffQueueLongPath(forge.ForgeModule):
    def __init__(self, name, stack_factor: int = 12):
        super().__init__(name)
        self.in0_mm_1 = forge.Parameter(16, 60 * 32, 60 * 32, requires_grad=False) 
        self.in1_mm_2 = forge.Parameter(1, 32 * 32, 1 * 32, requires_grad=False) 
        self.in1_mm_3 = forge.Parameter(1, 1 * 32, 32 * 32, requires_grad=False) 
        # in original graph in1_mm_3 has dimension 3 equal to 1 * 32. But mm_3 has broadcast on dimension 3 for 32.
        # pytorch doesn't allow for broadcast if dimension is greater than 1. So we can't broadcast here.
    def forward(self, act1, act2):
        # Longer path of fork join contains buffering queue,
        # which has to be taken into consideration when buffering fork-join.
        # fork,
        fork = forge.op.Concatenate("concatenate", act1, act2, axis=2)
        # right
        right = forge.op.Matmul("matmul_1", self.in0_mm_1, fork)
        forge.config._get_global_compiler_config().insert_queues = [("matmul_1", "matmul_2", 0)]
        right = forge.op.HStack("hstack", right)
        right = forge.op.Matmul("matmul_2", right, self.in1_mm_2)
        right = forge.op.Matmul("matmul_3", right, self.in1_mm_3)
        right = forge.op.HSlice("vslice", right, 16)
        # join
        join = forge.op.Subtract("join", fork, right)
        return join
# This test will hang on silicon if fork-join is not buffered properly. Longer path of fork join contains buffering queue,
# which has to be taken into consideration when buffering fork-join.
@pytest.mark.parametrize("format", [DataFormat.Bfp8_b, DataFormat.Float16_b], ids=["bfp8", "fp16"])
def test_fork_join_with_buff_queue_long_path(test_kind, test_device, format):
    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    verify_module(ForkJoinWithBuffQueueLongPath("test_fork_join_with_buff_queue_long_path"), [(1, 16, 40 * 32, 2 * 32), (1, 16, 20 * 32, 2 * 32)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, pcc=pcc, relative_atol=relative_atol,
                fp32_fallback=format))

class MultilevelForkJoin(forge.ForgeModule):
    def __init__(self, name,):
        super().__init__(name)
        self.add_parameter("stages.2.blocks.1.conv_mid.0.conv.weight", forge.Parameter(*(192, 768, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("stages.2.blocks.1.conv_mid.0.bn.weight", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("stages.2.blocks.1.conv_mid.0.bn.bias", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("stages.2.blocks.1.conv_mid.1.conv.weight", forge.Parameter(*(192, 192, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("stages.2.blocks.1.conv_mid.1.bn.weight", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("stages.2.blocks.1.conv_mid.1.bn.bias", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("stages.2.blocks.1.conv_mid.2.conv.weight", forge.Parameter(*(192, 192, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("stages.2.blocks.1.conv_mid.2.bn.weight", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("stages.2.blocks.1.conv_mid.2.bn.bias", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("stages.2.blocks.1.conv_mid.3.conv.weight", forge.Parameter(*(192, 192, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("stages.2.blocks.1.conv_mid.3.bn.weight", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("stages.2.blocks.1.conv_mid.3.bn.bias", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("stages.2.blocks.1.conv_mid.4.conv.weight", forge.Parameter(*(192, 192, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("stages.2.blocks.1.conv_mid.4.bn.weight", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("stages.2.blocks.1.conv_mid.4.bn.bias", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("stages.2.blocks.1.conv_concat.conv.weight", forge.Parameter(*(768, 1728, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32))

        self.add_constant("stages.2.blocks.1.conv_mid.0.bn.running_var")
        self.add_constant("stages.2.blocks.1.conv_mid.0.bn.running_mean")
        self.add_constant("stages.2.blocks.1.conv_mid.1.bn.running_var")
        self.add_constant("stages.2.blocks.1.conv_mid.1.bn.running_mean")
        self.add_constant("stages.2.blocks.1.conv_mid.2.bn.running_var")
        self.add_constant("stages.2.blocks.1.conv_mid.2.bn.running_mean")
        self.add_constant("stages.2.blocks.1.conv_mid.3.bn.running_var")
        self.add_constant("stages.2.blocks.1.conv_mid.3.bn.running_mean")
        self.add_constant("stages.2.blocks.1.conv_mid.4.bn.running_var")
        self.add_constant("stages.2.blocks.1.conv_mid.4.bn.running_mean")
        self.add_constant("const_67322")
        self.add_constant("const_68322")
        self.add_constant("const_69322")
        self.add_constant("const_70322")
        self.add_constant("const_71322")
        self.add_constant("const_72322")
        self.add_constant("const_73322")
        self.add_constant("const_74322")
        self.add_constant("const_75322")
        self.add_constant("const_76322")
        self.add_constant("const_77322")
        self.add_constant("const_78322")
        self.add_constant("const_79322")
        self.add_constant("const_80322")
        self.add_constant("const_81322")

        self.set_constant("stages.2.blocks.1.conv_mid.0.bn.running_var", torch.rand(1, 192))
        self.set_constant("stages.2.blocks.1.conv_mid.0.bn.running_mean", torch.rand(1, 192))
        self.set_constant("stages.2.blocks.1.conv_mid.1.bn.running_var", torch.rand(1, 192))
        self.set_constant("stages.2.blocks.1.conv_mid.1.bn.running_mean", torch.rand(1, 192))
        self.set_constant("stages.2.blocks.1.conv_mid.2.bn.running_var", torch.rand(1, 192))
        self.set_constant("stages.2.blocks.1.conv_mid.2.bn.running_mean", torch.rand(1, 192))
        self.set_constant("stages.2.blocks.1.conv_mid.3.bn.running_var", torch.rand(1, 192))
        self.set_constant("stages.2.blocks.1.conv_mid.3.bn.running_mean", torch.rand(1, 192))
        self.set_constant("stages.2.blocks.1.conv_mid.4.bn.running_var", torch.rand(1, 192))
        self.set_constant("stages.2.blocks.1.conv_mid.4.bn.running_mean", torch.rand(1, 192))
        self.set_constant("const_67322", torch.rand(1, 1))
        self.set_constant("const_68322", torch.rand(1, 1))
        self.set_constant("const_69322", torch.rand(1, 1))
        self.set_constant("const_70322", torch.rand(1, 1))
        self.set_constant("const_71322", torch.rand(1, 1))
        self.set_constant("const_72322", torch.rand(1, 1))
        self.set_constant("const_73322", torch.rand(1, 1))
        self.set_constant("const_74322", torch.rand(1, 1))
        self.set_constant("const_75322", torch.rand(1, 1))
        self.set_constant("const_76322", torch.rand(1, 1))
        self.set_constant("const_77322", torch.rand(1, 1))
        self.set_constant("const_78322", torch.rand(1, 1))
        self.set_constant("const_79322", torch.rand(1, 1))
        self.set_constant("const_80322", torch.rand(1, 1))
        self.set_constant("const_81322", torch.rand(1, 1))

    def forward(self, act_0):

        conv2d_586 = forge.op.Conv2d("", act_0, self.get_parameter("stages.2.blocks.1.conv_mid.0.conv.weight"), stride=[1, 1], padding=[1, 1, 1, 1], dilation=1, groups=1, channel_last=0)
        add_589 = forge.op.Add("", self.get_constant("stages.2.blocks.1.conv_mid.0.bn.running_var"), self.get_constant("const_68322"))
        sqrt_590 = forge.op.Sqrt("", add_589)
        reciprocal_591 = forge.op.Reciprocal("", sqrt_590)
        multiply_592 = forge.op.Multiply("", self.get_constant("const_67322"), reciprocal_591)
        multiply_593 = forge.op.Multiply("", multiply_592, self.get_parameter("stages.2.blocks.1.conv_mid.0.bn.weight"))
        reshape_594 = forge.op.Reshape("", multiply_593, shape=(192, 1, 1))
        multiply_595 = forge.op.Multiply("", conv2d_586, reshape_594)
        multiply_597 = forge.op.Multiply("", self.get_constant("stages.2.blocks.1.conv_mid.0.bn.running_mean"), self.get_constant("const_69322"))
        multiply_598 = forge.op.Multiply("", multiply_597, multiply_593)
        add_599 = forge.op.Add("", multiply_598, self.get_parameter("stages.2.blocks.1.conv_mid.0.bn.bias"))
        reshape_600 = forge.op.Reshape("", add_599, shape=(192, 1, 1))
        add_601 = forge.op.Add("", multiply_595, reshape_600)
        relu_602 = forge.op.Relu("", add_601)
        conv2d_603 = forge.op.Conv2d("", relu_602, self.get_parameter("stages.2.blocks.1.conv_mid.1.conv.weight"), stride=[1, 1], padding=[1, 1, 1, 1], dilation=1, groups=1, channel_last=0)
        add_606 = forge.op.Add("", self.get_constant("stages.2.blocks.1.conv_mid.1.bn.running_var"), self.get_constant("const_71322"))
        sqrt_607 = forge.op.Sqrt("", add_606)
        reciprocal_608 = forge.op.Reciprocal("", sqrt_607)
        multiply_609 = forge.op.Multiply("", self.get_constant("const_70322"), reciprocal_608)
        multiply_610 = forge.op.Multiply("", multiply_609, self.get_parameter("stages.2.blocks.1.conv_mid.1.bn.weight"))
        reshape_611 = forge.op.Reshape("", multiply_610, shape=(192, 1, 1))
        multiply_612 = forge.op.Multiply("", conv2d_603, reshape_611)
        multiply_614 = forge.op.Multiply("", self.get_constant("stages.2.blocks.1.conv_mid.1.bn.running_mean"), self.get_constant("const_72322"))
        multiply_615 = forge.op.Multiply("", multiply_614, multiply_610)
        add_616 = forge.op.Add("", multiply_615, self.get_parameter("stages.2.blocks.1.conv_mid.1.bn.bias"))
        reshape_617 = forge.op.Reshape("", add_616, shape=(192, 1, 1))
        add_618 = forge.op.Add("", multiply_612, reshape_617)
        relu_619 = forge.op.Relu("", add_618)
        conv2d_620 = forge.op.Conv2d("", relu_619, self.get_parameter("stages.2.blocks.1.conv_mid.2.conv.weight"), stride=[1, 1], padding=[1, 1, 1, 1], dilation=1, groups=1, channel_last=0)
        add_623 = forge.op.Add("", self.get_constant("stages.2.blocks.1.conv_mid.2.bn.running_var"), self.get_constant("const_74322"))
        sqrt_624 = forge.op.Sqrt("", add_623)
        reciprocal_625 = forge.op.Reciprocal("", sqrt_624)
        multiply_626 = forge.op.Multiply("", self.get_constant("const_73322"), reciprocal_625)
        multiply_627 = forge.op.Multiply("", multiply_626, self.get_parameter("stages.2.blocks.1.conv_mid.2.bn.weight"))
        reshape_628 = forge.op.Reshape("", multiply_627, shape=(192, 1, 1))
        multiply_629 = forge.op.Multiply("", conv2d_620, reshape_628)
        multiply_631 = forge.op.Multiply("", self.get_constant("stages.2.blocks.1.conv_mid.2.bn.running_mean"), self.get_constant("const_75322"))
        multiply_632 = forge.op.Multiply("", multiply_631, multiply_627)
        add_633 = forge.op.Add("", multiply_632, self.get_parameter("stages.2.blocks.1.conv_mid.2.bn.bias"))
        reshape_634 = forge.op.Reshape("", add_633, shape=(192, 1, 1))
        add_635 = forge.op.Add("", multiply_629, reshape_634)
        relu_636 = forge.op.Relu("", add_635)
        conv2d_637 = forge.op.Conv2d("", relu_636, self.get_parameter("stages.2.blocks.1.conv_mid.3.conv.weight"), stride=[1, 1], padding=[1, 1, 1, 1], dilation=1, groups=1, channel_last=0)
        add_640 = forge.op.Add("", self.get_constant("stages.2.blocks.1.conv_mid.3.bn.running_var"), self.get_constant("const_77322"))
        sqrt_641 = forge.op.Sqrt("", add_640)
        reciprocal_642 = forge.op.Reciprocal("", sqrt_641)
        multiply_643 = forge.op.Multiply("", self.get_constant("const_76322"), reciprocal_642)
        multiply_644 = forge.op.Multiply("", multiply_643, self.get_parameter("stages.2.blocks.1.conv_mid.3.bn.weight"))
        reshape_645 = forge.op.Reshape("", multiply_644, shape=(192, 1, 1))
        multiply_646 = forge.op.Multiply("", conv2d_637, reshape_645)
        multiply_648 = forge.op.Multiply("", self.get_constant("stages.2.blocks.1.conv_mid.3.bn.running_mean"), self.get_constant("const_78322"))
        multiply_649 = forge.op.Multiply("", multiply_648, multiply_644)
        add_650 = forge.op.Add("", multiply_649, self.get_parameter("stages.2.blocks.1.conv_mid.3.bn.bias"))
        reshape_651 = forge.op.Reshape("", add_650, shape=(192, 1, 1))
        add_652 = forge.op.Add("", multiply_646, reshape_651)
        relu_653 = forge.op.Relu("", add_652)
        conv2d_654 = forge.op.Conv2d("", relu_653, self.get_parameter("stages.2.blocks.1.conv_mid.4.conv.weight"), stride=[1, 1], padding=[1, 1, 1, 1], dilation=1, groups=1, channel_last=0)
        add_657 = forge.op.Add("", self.get_constant("stages.2.blocks.1.conv_mid.4.bn.running_var"), self.get_constant("const_80322"))
        sqrt_658 = forge.op.Sqrt("", add_657)
        reciprocal_659 = forge.op.Reciprocal("", sqrt_658)
        multiply_660 = forge.op.Multiply("", self.get_constant("const_79322"), reciprocal_659)
        multiply_661 = forge.op.Multiply("", multiply_660, self.get_parameter("stages.2.blocks.1.conv_mid.4.bn.weight"))
        reshape_662 = forge.op.Reshape("", multiply_661, shape=(192, 1, 1))
        multiply_663 = forge.op.Multiply("", conv2d_654, reshape_662)
        multiply_665 = forge.op.Multiply("", self.get_constant("stages.2.blocks.1.conv_mid.4.bn.running_mean"), self.get_constant("const_81322"))
        multiply_666 = forge.op.Multiply("", multiply_665, multiply_661)
        add_667 = forge.op.Add("", multiply_666, self.get_parameter("stages.2.blocks.1.conv_mid.4.bn.bias"))
        reshape_668 = forge.op.Reshape("", add_667, shape=(192, 1, 1))
        add_669 = forge.op.Add("", multiply_663, reshape_668)
        relu_670 = forge.op.Relu("", add_669)
        concatenate_671 = forge.op.Concatenate("", act_0, relu_602, relu_619, relu_636, relu_653, relu_670, axis=-3)
        conv2d_672 = forge.op.Conv2d("", concatenate_671, self.get_parameter("stages.2.blocks.1.conv_concat.conv.weight"), stride=[1, 1], padding=[0, 0, 0, 0], dilation=1, groups=1, channel_last=0)

        return conv2d_672

# This test will hang on silicon if fork-join is not buffered properly. This test is from vovnet_v2 benchmark.
# This test will hang without fork-join multilevel feature fec3b1879941dde87fa7f1d460ba5ff1bbb751f4
@pytest.mark.parametrize("format", [DataFormat.Bfp8_b, DataFormat.Float16_b], ids=["bfp8", "fp16"])
def test_multilevel_fork_join_vovnet(test_kind, test_device, format):
    if test_kind.is_training():
        pytest.skip()
    try:
        import os
        os.environ["FORGE_MAXIMIZE_SPARSE_UBLOCK"] = "1"
        os.environ["FORGE_RIBBON2"] = "1"

        compiler_cfg = forge.config._get_global_compiler_config()
        compiler_cfg.balancer_policy = "Ribbon"
        compiler_cfg.default_df_override = format
        # Op overrides
        forge.config.override_op_size("conv2d_0.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (1, 4))
        forge.config.override_op_size("conv2d_14.dc.matmul.11", (1, 2))
        forge.config.override_op_size("conv2d_14.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (1, 3))
        forge.config.override_op_size("conv2d_28.dc.matmul.11", (1, 2))
        forge.config.override_op_size("conv2d_28.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (1, 3))
        forge.config.override_op_size("conv2d_42.dc.matmul.11", (1, 2))
        forge.config.override_op_size("conv2d_42.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (1, 3))
        forge.config.override_op_size("conv2d_56.dc.matmul.11", (1, 2))
        forge.config.override_op_size("conv2d_56.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (1, 3))
        forge.config.override_op_size("concatenate_70.dc.concatenate.0", (1, 1))

        relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
        verify_module(MultilevelForkJoin("test_multilevel_fork_join_vovnet"),[(1, 768, 14, 14)],
                VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, pcc=pcc, relative_atol=relative_atol,
                    fp32_fallback=format))
    finally:
        # unset env variables
        os.environ.pop('FORGE_MAXIMIZE_SPARSE_UBLOCK', None)
        os.environ.pop('FORGE_RIBBON2', None)

class BertGeluFork(forge.ForgeModule):

    def __init__(self, name, seq_len=128, hidden_dim=784):
        super().__init__(name)
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.weights1 = forge.Parameter(hidden_dim, hidden_dim*4);
        self.weights2 = forge.Parameter(hidden_dim*4, hidden_dim);

    def forward(self, act):

        # fork
        fork = forge.op.Buffer("fork", act)

        # right
        right = forge.op.Matmul("ff1", fork, self.weights1)
        right = forge.op.Gelu("gelu", right)
        right = forge.op.Matmul("ff2", right, self.weights2)

        # join
        join = forge.op.Add("join", fork, right)
        return join

@pytest.mark.parametrize("format", [DataFormat.Bfp8_b, DataFormat.Float16_b], ids=["bfp8", "fp16"])
@pytest.mark.skip(reason="too slow for CI")
def test_bert_gelu_fork(test_kind, test_device, format):
    microbatch_count = 256
    seq_len = 128
    hidden_dim = 768

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    forge.config._get_global_compiler_config().performance_trace = forge.config.PerfTraceLevel.VERBOSE
    verify_module(BertGeluFork("bert_gelu_fork", seq_len, hidden_dim), [(microbatch_count, seq_len, hidden_dim)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, pcc=pcc, relative_atol=relative_atol,
                fp32_fallback=format), params_centered_on_zero=True)

class BertReduceFork(forge.ForgeModule):

    def __init__(self, name, seq_len=128, hidden_dim=784):
        super().__init__(name)
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.weights1 = forge.Parameter(seq_len, hidden_dim);

    def forward(self, act):

        # fork
        fork = forge.op.Buffer("fork", act)

        # right
        right = forge.op.Add("add", fork, self.weights1)
        right = forge.op.ReduceAvg("reduce", right, dim=-1)

        # join
        join = forge.op.Add("join", fork, right)
        return join

@pytest.mark.parametrize("format", [DataFormat.Bfp8_b, DataFormat.Float16_b], ids=["bfp8", "fp16"])
@pytest.mark.skip(reason="too slow for CI")
def test_bert_reduce_fork(test_kind, test_device, format):
    microbatch_count = 256
    seq_len = 384
    hidden_dim = 1024

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    forge.config._get_global_compiler_config().performance_trace = forge.config.PerfTraceLevel.VERBOSE
    verify_module(BertReduceFork("bert_reduce_fork", seq_len, hidden_dim), [(microbatch_count, seq_len, hidden_dim)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, pcc=pcc, relative_atol=relative_atol,
                fp32_fallback=format), params_centered_on_zero=True)


class PipelineStuck(forge.ForgeModule):

    def __init__(self, name):
        super().__init__(name)

    def forward(self, act):

        # fork
        #act = forge.op.ReduceAvg("reduce", act, dim=-1)
        act = forge.op.Sqrt("sqrt", act)
        act = forge.op.Exp("exp", act)
        act = forge.op.Buffer("nop2", act)

        return act

@pytest.mark.parametrize("format", [DataFormat.Bfp8_b, DataFormat.Float16_b], ids=["bfp8", "fp16"])
@pytest.mark.skip(reason="too slow for CI")
def test_pipeline_stuck(test_kind, test_device, format):
    microbatch_count = 256
    seq_len = 128
    hidden_dim = 768

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    forge.config._get_global_compiler_config().performance_trace = forge.config.PerfTraceLevel.VERBOSE
    verify_module(PipelineStuck("pipeline_stuck"), [(microbatch_count, seq_len, hidden_dim)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, pcc=pcc, relative_atol=relative_atol,
                fp32_fallback=format), params_centered_on_zero=True)


class NestedForks(forge.ForgeModule):

    def __init__(self, name):
        super().__init__(name)

    def forward(self, act):

        # main fork
        fork = forge.op.Buffer("main_fork", act)

        left_1 = forge.op.Buffer("left_1", fork)
        left_2 = forge.op.Buffer("left_2", left_1)
        fork_2 = forge.op.Buffer("fork_2", left_2)
        right_2_1 = forge.op.Buffer("right_2_1", fork_2)
        join_2 = forge.op.Add("join_2", fork_2, right_2_1)

        right_1 = forge.op.Buffer("right_1", fork)
        join_3 = forge.op.Add("join_3", right_1, join_2)

        left_4 = forge.op.Buffer("left_4", join_3)

        join = forge.op.Add("join", fork, left_4)

        return join

def test_nested_forks(test_kind, test_device):
    microbatch_count = 1
    seq_len = 128
    hidden_dim = 768

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    #forge.config._get_global_compiler_config().performance_trace = forge.config.PerfTraceLevel.VERBOSE
    verify_module(NestedForks("netsted_forks"), [(microbatch_count, seq_len, hidden_dim)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, pcc=pcc, relative_atol=relative_atol), params_centered_on_zero=True)

class YoloV3ForkJoin(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter("backbone.base.conv.weight", forge.Parameter(*(32, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("backbone.base.bn.weight", forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("backbone.base.bn.bias", forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("backbone.darknet_0.0.conv.weight", forge.Parameter(*(64, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("backbone.darknet_0.0.bn.weight", forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("backbone.darknet_0.0.bn.bias", forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("backbone.darknet_0.1.conv1.conv.weight", forge.Parameter(*(32, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("backbone.darknet_0.1.conv1.bn.weight", forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("backbone.darknet_0.1.conv1.bn.bias", forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("backbone.darknet_0.1.conv2.conv.weight", forge.Parameter(*(64, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("backbone.darknet_0.1.conv2.bn.weight", forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("backbone.darknet_0.1.conv2.bn.bias", forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_constant("backbone.base.bn.running_var", shape=(32,))
        self.set_constant("backbone.base.bn.running_var", torch.rand(32, ))
        self.add_constant("backbone.base.bn.running_mean", shape=(32,))
        self.set_constant("backbone.base.bn.running_mean", torch.rand(32, ))
        self.add_constant("backbone.darknet_0.0.bn.running_var", shape=(64,))
        self.set_constant("backbone.darknet_0.0.bn.running_var", torch.rand(64, ))
        self.add_constant("backbone.darknet_0.0.bn.running_mean", shape=(64,))
        self.set_constant("backbone.darknet_0.0.bn.running_mean", torch.rand(64, ))
        self.add_constant("backbone.darknet_0.1.conv1.bn.running_var", shape=(32,))
        self.set_constant("backbone.darknet_0.1.conv1.bn.running_var", torch.rand(32, ))
        self.add_constant("backbone.darknet_0.1.conv1.bn.running_mean", shape=(32,))
        self.set_constant("backbone.darknet_0.1.conv1.bn.running_mean", torch.rand(32, ))
        self.add_constant("backbone.darknet_0.1.conv2.bn.running_var", shape=(64,))
        self.set_constant("backbone.darknet_0.1.conv2.bn.running_var", torch.rand(64, ))
        self.add_constant("backbone.darknet_0.1.conv2.bn.running_mean", shape=(64,))
        self.set_constant("backbone.darknet_0.1.conv2.bn.running_mean", torch.rand(64, ))
        self.add_constant("const_0578", shape=(1, 1))
        self.set_constant("const_0578", torch.rand(1, 1))
        self.add_constant("const_1578", shape=(1, 1))
        self.set_constant("const_1578", torch.rand(1, 1))
        self.add_constant("const_2578", shape=(1, 1))
        self.set_constant("const_2578", torch.rand(1, 1))
        self.add_constant("const_3578", shape=(1, 1))
        self.set_constant("const_3578", torch.rand(1, 1))
        self.add_constant("const_4578", shape=(1, 1))
        self.set_constant("const_4578", torch.rand(1, 1))
        self.add_constant("const_5578", shape=(1, 1))
        self.set_constant("const_5578", torch.rand(1, 1))
        self.add_constant("const_6578", shape=(1, 1))
        self.set_constant("const_6578", torch.rand(1, 1))
        self.add_constant("const_7578", shape=(1, 1))
        self.set_constant("const_7578", torch.rand(1, 1))
        self.add_constant("const_8578", shape=(1, 1))
        self.set_constant("const_8578", torch.rand(1, 1))
        self.add_constant("const_9578", shape=(1, 1))
        self.set_constant("const_9578", torch.rand(1, 1))
        self.add_constant("const_10578", shape=(1, 1))
        self.set_constant("const_10578", torch.rand(1, 1))
        self.add_constant("const_11578", shape=(1, 1))
        self.set_constant("const_11578", torch.rand(1, 1))

    # Input shapes:
    # x_1 -> (1, 3, 512, 512)
    def forward(self, x_1):
        conv2d_367 = forge.op.Conv2d("conv2d_0", x_1, self.get_parameter("backbone.base.conv.weight"), stride=[1, 1], padding=[1, 1, 1, 1], dilation=1, groups=1, channel_last=0)
        add_370 = forge.op.Add("add_1", self.get_constant("backbone.base.bn.running_var"), self.get_constant("const_1578"))
        sqrt_371 = forge.op.Sqrt("sqrt_2", add_370)
        reciprocal_372 = forge.op.Reciprocal("reciprocal_3", sqrt_371)
        multiply_373 = forge.op.Multiply("multiply_4", self.get_constant("const_0578"), reciprocal_372)
        multiply_374 = forge.op.Multiply("multiply_5", multiply_373, self.get_parameter("backbone.base.bn.weight"))
        reshape_375 = forge.op.Reshape("reshape_6", multiply_374, shape=(32, 1, 1))
        multiply_376 = forge.op.Multiply("multiply_7", conv2d_367, reshape_375)
        multiply_378 = forge.op.Multiply("multiply_8", self.get_constant("backbone.base.bn.running_mean"), self.get_constant("const_2578"))
        multiply_379 = forge.op.Multiply("multiply_9", multiply_378, multiply_374)
        add_380 = forge.op.Add("add_10", multiply_379, self.get_parameter("backbone.base.bn.bias"))
        reshape_381 = forge.op.Reshape("reshape_11", add_380, shape=(32, 1, 1))
        add_382 = forge.op.Add("add_12", multiply_376, reshape_381)
        leaky_relu_383 = forge.op.LeakyRelu("leaky_relu_13", add_382, alpha=0.10000000000000001)
        conv2d_384 = forge.op.Conv2d("conv2d_14", leaky_relu_383, self.get_parameter("backbone.darknet_0.0.conv.weight"), stride=[2, 2], padding=[1, 1, 1, 1], dilation=1, groups=1, channel_last=0)
        add_387 = forge.op.Add("add_15", self.get_constant("backbone.darknet_0.0.bn.running_var"), self.get_constant("const_4578"))
        sqrt_388 = forge.op.Sqrt("sqrt_16", add_387)
        reciprocal_389 = forge.op.Reciprocal("reciprocal_17", sqrt_388)
        multiply_390 = forge.op.Multiply("multiply_18", self.get_constant("const_3578"), reciprocal_389)
        multiply_391 = forge.op.Multiply("multiply_19", multiply_390, self.get_parameter("backbone.darknet_0.0.bn.weight"))
        reshape_392 = forge.op.Reshape("reshape_20", multiply_391, shape=(64, 1, 1))
        multiply_393 = forge.op.Multiply("multiply_21", conv2d_384, reshape_392)
        multiply_395 = forge.op.Multiply("multiply_22", self.get_constant("backbone.darknet_0.0.bn.running_mean"), self.get_constant("const_5578"))
        multiply_396 = forge.op.Multiply("multiply_23", multiply_395, multiply_391)
        add_397 = forge.op.Add("add_24", multiply_396, self.get_parameter("backbone.darknet_0.0.bn.bias"))
        reshape_398 = forge.op.Reshape("reshape_25", add_397, shape=(64, 1, 1))
        add_399 = forge.op.Add("add_26", multiply_393, reshape_398)
        leaky_relu_400 = forge.op.LeakyRelu("leaky_relu_27", add_399, alpha=0.10000000000000001)
        conv2d_401 = forge.op.Conv2d("conv2d_28", leaky_relu_400, self.get_parameter("backbone.darknet_0.1.conv1.conv.weight"), stride=[1, 1], padding=[0, 0, 0, 0], dilation=1, groups=1, channel_last=0)
        add_404 = forge.op.Add("add_29", self.get_constant("backbone.darknet_0.1.conv1.bn.running_var"), self.get_constant("const_7578"))
        sqrt_405 = forge.op.Sqrt("sqrt_30", add_404)
        reciprocal_406 = forge.op.Reciprocal("reciprocal_31", sqrt_405)
        multiply_407 = forge.op.Multiply("multiply_32", self.get_constant("const_6578"), reciprocal_406)
        multiply_408 = forge.op.Multiply("multiply_33", multiply_407, self.get_parameter("backbone.darknet_0.1.conv1.bn.weight"))
        reshape_409 = forge.op.Reshape("reshape_34", multiply_408, shape=(32, 1, 1))
        multiply_410 = forge.op.Multiply("multiply_35", conv2d_401, reshape_409)
        multiply_412 = forge.op.Multiply("multiply_36", self.get_constant("backbone.darknet_0.1.conv1.bn.running_mean"), self.get_constant("const_8578"))
        multiply_413 = forge.op.Multiply("multiply_37", multiply_412, multiply_408)
        add_414 = forge.op.Add("add_38", multiply_413, self.get_parameter("backbone.darknet_0.1.conv1.bn.bias"))
        reshape_415 = forge.op.Reshape("reshape_39", add_414, shape=(32, 1, 1))
        add_416 = forge.op.Add("add_40", multiply_410, reshape_415)
        leaky_relu_417 = forge.op.LeakyRelu("leaky_relu_41", add_416, alpha=0.10000000000000001)
        conv2d_418 = forge.op.Conv2d("conv2d_42", leaky_relu_417, self.get_parameter("backbone.darknet_0.1.conv2.conv.weight"), stride=[1, 1], padding=[1, 1, 1, 1], dilation=1, groups=1, channel_last=0)
        add_421 = forge.op.Add("add_43", self.get_constant("backbone.darknet_0.1.conv2.bn.running_var"), self.get_constant("const_10578"))
        sqrt_422 = forge.op.Sqrt("sqrt_44", add_421)
        reciprocal_423 = forge.op.Reciprocal("reciprocal_45", sqrt_422)
        multiply_424 = forge.op.Multiply("multiply_46", self.get_constant("const_9578"), reciprocal_423)
        multiply_425 = forge.op.Multiply("multiply_47", multiply_424, self.get_parameter("backbone.darknet_0.1.conv2.bn.weight"))
        reshape_426 = forge.op.Reshape("reshape_48", multiply_425, shape=(64, 1, 1))
        multiply_427 = forge.op.Multiply("multiply_49", conv2d_418, reshape_426)
        multiply_429 = forge.op.Multiply("multiply_50", self.get_constant("backbone.darknet_0.1.conv2.bn.running_mean"), self.get_constant("const_11578"))
        multiply_430 = forge.op.Multiply("multiply_51", multiply_429, multiply_425)
        add_431 = forge.op.Add("add_52", multiply_430, self.get_parameter("backbone.darknet_0.1.conv2.bn.bias"))
        reshape_432 = forge.op.Reshape("reshape_53", add_431, shape=(64, 1, 1))
        add_433 = forge.op.Add("add_54", multiply_427, reshape_432)
        leaky_relu_434 = forge.op.LeakyRelu("leaky_relu_55", add_433, alpha=0.10000000000000001)
        add_435 = forge.op.Add("add_56", leaky_relu_434, leaky_relu_400)
        reshape_436 = forge.op.Reshape("reshape_final", add_435, shape=(1, 1, 64, 65536))
        return reshape_436

    @staticmethod
    def add_op_overrides():
        forge.config.override_op_size("_fused_op_2", (2, 2))
        forge.config.override_t_stream_shape("_fused_op_2", (128, 1))
        forge.config.override_t_stream_dir("_fused_op_2", "r")
        forge.config.override_op_size("conv2d_42.dc.conv2d.1.dc.matmul.11", (2, 2))
        forge.config.override_t_stream_shape("conv2d_42.dc.conv2d.1.dc.matmul.11", (128, 1))
        forge.config.override_t_stream_dir("conv2d_42.dc.conv2d.1.dc.matmul.11", "r")
        forge.config.override_u_kt("conv2d_42.dc.conv2d.1.dc.matmul.11", 1)
        forge.config.override_op_size("conv2d_42.dc.conv2d.3.dc.matmul.11", (2, 2))
        forge.config.override_t_stream_shape("conv2d_42.dc.conv2d.3.dc.matmul.11", (128, 1))
        forge.config.override_t_stream_dir("conv2d_42.dc.conv2d.3.dc.matmul.11", "r")
        forge.config.override_u_kt("conv2d_42.dc.conv2d.3.dc.matmul.11", 1)
        forge.config.override_op_size("conv2d_42.dc.conv2d.5.dc.matmul.11", (2, 2))
        forge.config.override_t_stream_shape("conv2d_42.dc.conv2d.5.dc.matmul.11", (128, 1))
        forge.config.override_t_stream_dir("conv2d_42.dc.conv2d.5.dc.matmul.11", "r")
        forge.config.override_u_kt("conv2d_42.dc.conv2d.5.dc.matmul.11", 1)
        forge.config.override_op_size("_fused_op_1", (2, 2))
        forge.config.override_t_stream_shape("_fused_op_1", (128, 1))
        forge.config.override_t_stream_dir("_fused_op_1", "r")
        forge.config.override_op_size("conv2d_42.dc.conv2d.1.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (2, 1))
        forge.config.override_t_stream_shape("conv2d_42.dc.conv2d.1.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (128, 1))
        forge.config.override_t_stream_dir("conv2d_42.dc.conv2d.1.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", "rz")
        forge.config.override_op_size("conv2d_42.dc.conv2d.3.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (2, 1))
        forge.config.override_t_stream_shape("conv2d_42.dc.conv2d.3.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (128, 1))
        forge.config.override_t_stream_dir("conv2d_42.dc.conv2d.3.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", "rz")
        forge.config.override_op_size("conv2d_42.dc.conv2d.5.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (2, 1))
        forge.config.override_t_stream_shape("conv2d_42.dc.conv2d.5.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (128, 1))
        forge.config.override_t_stream_dir("conv2d_42.dc.conv2d.5.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", "rz")
        forge.config.override_op_size("conv2d_14.dc.conv2d.1.dc.matmul.11", (2, 2))
        forge.config.override_t_stream_shape("conv2d_14.dc.conv2d.1.dc.matmul.11", (128, 1))
        forge.config.override_t_stream_dir("conv2d_14.dc.conv2d.1.dc.matmul.11", "r")
        forge.config.override_u_kt("conv2d_14.dc.conv2d.1.dc.matmul.11", 1)
        forge.config.override_op_size("conv2d_14.dc.conv2d.3.dc.matmul.11", (2, 2))
        forge.config.override_t_stream_shape("conv2d_14.dc.conv2d.3.dc.matmul.11", (128, 1))
        forge.config.override_t_stream_dir("conv2d_14.dc.conv2d.3.dc.matmul.11", "r")
        forge.config.override_u_kt("conv2d_14.dc.conv2d.3.dc.matmul.11", 1)
        forge.config.override_op_size("conv2d_14.dc.conv2d.5.dc.matmul.11", (2, 2))
        forge.config.override_t_stream_shape("conv2d_14.dc.conv2d.5.dc.matmul.11", (128, 1))
        forge.config.override_t_stream_dir("conv2d_14.dc.conv2d.5.dc.matmul.11", "r")
        forge.config.override_u_kt("conv2d_14.dc.conv2d.5.dc.matmul.11", 1)
        forge.config.override_op_size("leaky_relu_41", (2, 1))
        forge.config.override_t_stream_shape("leaky_relu_41", (128, 1))
        forge.config.override_t_stream_dir("leaky_relu_41", "r")
        forge.config.override_op_size("conv2d_14.dc.conv2d.1.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (2, 1))
        forge.config.override_t_stream_shape("conv2d_14.dc.conv2d.1.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (128, 1))
        forge.config.override_t_stream_dir("conv2d_14.dc.conv2d.1.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", "rz")
        forge.config.override_op_size("conv2d_14.dc.conv2d.3.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (2, 1))
        forge.config.override_t_stream_shape("conv2d_14.dc.conv2d.3.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (128, 1))
        forge.config.override_t_stream_dir("conv2d_14.dc.conv2d.3.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", "rz")
        forge.config.override_op_size("conv2d_14.dc.conv2d.5.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (2, 1))
        forge.config.override_t_stream_shape("conv2d_14.dc.conv2d.5.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (128, 1))
        forge.config.override_t_stream_dir("conv2d_14.dc.conv2d.5.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", "rz")
        forge.config.override_op_size("conv2d_28.dc.matmul.8", (2, 1))
        forge.config.override_t_stream_shape("conv2d_28.dc.matmul.8", (128, 1))
        forge.config.override_t_stream_dir("conv2d_28.dc.matmul.8", "r")
        forge.config.override_u_kt("conv2d_28.dc.matmul.8", 2)
        forge.config.override_op_size("_fused_op_0", (2, 1))
        forge.config.override_t_stream_shape("_fused_op_0", (256, 1))
        forge.config.override_t_stream_dir("_fused_op_0", "r")
        forge.config.override_op_size("conv2d_0.dc.conv2d.1.dc.matmul.11", (2, 1))
        forge.config.override_t_stream_shape("conv2d_0.dc.conv2d.1.dc.matmul.11", (256, 1))
        forge.config.override_t_stream_dir("conv2d_0.dc.conv2d.1.dc.matmul.11", "r")
        forge.config.override_u_kt("conv2d_0.dc.conv2d.1.dc.matmul.11", 1)
        forge.config.override_op_size("conv2d_0.dc.conv2d.3.dc.matmul.11", (2, 1))
        forge.config.override_t_stream_shape("conv2d_0.dc.conv2d.3.dc.matmul.11", (256, 1))
        forge.config.override_t_stream_dir("conv2d_0.dc.conv2d.3.dc.matmul.11", "r")
        forge.config.override_u_kt("conv2d_0.dc.conv2d.3.dc.matmul.11", 1)
        forge.config.override_op_size("conv2d_0.dc.conv2d.5.dc.matmul.11", (2, 1))
        forge.config.override_t_stream_shape("conv2d_0.dc.conv2d.5.dc.matmul.11", (256, 1))
        forge.config.override_t_stream_dir("conv2d_0.dc.conv2d.5.dc.matmul.11", "r")
        forge.config.override_u_kt("conv2d_0.dc.conv2d.5.dc.matmul.11", 1)
        forge.config.override_op_size("conv2d_0.dc.conv2d.1.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (2, 1))
        forge.config.override_t_stream_shape("conv2d_0.dc.conv2d.1.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (256, 1))
        forge.config.override_t_stream_dir("conv2d_0.dc.conv2d.1.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", "rz")
        forge.config.override_op_size("conv2d_0.dc.conv2d.3.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (2, 1))
        forge.config.override_t_stream_shape("conv2d_0.dc.conv2d.3.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (256, 1))
        forge.config.override_t_stream_dir("conv2d_0.dc.conv2d.3.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", "rz")
        forge.config.override_op_size("conv2d_0.dc.conv2d.5.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (2, 1))
        forge.config.override_t_stream_shape("conv2d_0.dc.conv2d.5.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (256, 1))
        forge.config.override_t_stream_dir("conv2d_0.dc.conv2d.5.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", "rz")

def test_fork_join_yolo_v3(test_kind, test_device):
    """
    This test is extracted from yolo_v3 benchmark model.

    Fork-join which causes hang is the one from _fused_op_1 to _fused_op_2.
    FORGE_FORK_JOIN_EXPAND_OUTPUT_BUFFERS=1 fixes the hang.
    """

    if test_kind.is_training():
        pytest.skip("Skipping training due to op overrides.")

    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = DataFormat.Float16_b
    compiler_cfg.enable_auto_transposing_placement = True

    YoloV3ForkJoin.add_op_overrides()
    import os
    os.environ["FORGE_RIBBON2"] = "1"
    os.environ["FORGE_FORCE_SEQUENTIAL"] = "1" # TODO: Figure out why this is needed, segfaults otherwise: tenstorrent/forge#1935
    os.environ["FORGE_OVERRIDE_INPUT_QUEUE_ENTRIES"] = "32"
    os.environ["FORGE_MAXIMIZE_SPARSE_UBLOCK"] = "1"
    os.environ["FORGE_DISABLE_CAP_SPARSE_MM_FIDELITY"] = "1"
    os.environ["FORGE_DISABLE_EXPLICIT_DRAM_IO"] = "1"

    # Fixes hang
    os.environ["FORGE_FORK_JOIN_EXPAND_OUTPUT_BUFFERS"] = "1"
    
    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    verify_module(YoloV3ForkJoin("test_fork_join_yolo_v3"), [(32, 3, 512, 512)],
                  VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, pcc=pcc, relative_atol=relative_atol))

class HRNetForkJoin(forge.ForgeModule):

    def __init__(self, name):
        super().__init__(name)
        self.add_parameter("features.init_block.conv1.conv.weight", forge.Parameter(*(64, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("features.init_block.conv1.bn.weight", forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("features.init_block.conv1.bn.bias", forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("features.init_block.conv2.conv.weight", forge.Parameter(*(64, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("features.init_block.subblocks.block1.body.conv1.conv.weight", forge.Parameter(*(64, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("features.init_block.subblocks.block1.body.conv2.conv.weight", forge.Parameter(*(64, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("features.init_block.subblocks.block1.body.conv3.conv.weight", forge.Parameter(*(256, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("features.init_block.subblocks.block1.identity_conv.conv.weight", forge.Parameter(*(256, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("features.init_block.subblocks.block2.body.conv1.conv.weight", forge.Parameter(*(64, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("bla", forge.Parameter(*(256, 56, 56), requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("bias1", forge.Parameter(64, requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("bias2", forge.Parameter(64, requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("bias3", forge.Parameter(64, requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("bias4", forge.Parameter(256, requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_parameter("bias5", forge.Parameter(256, requires_grad=True, dev_data_format=forge.DataFormat.Float32))
        self.add_constant("features.init_block.conv1.bn.running_var")
        self.add_constant("features.init_block.conv1.bn.running_mean")
        self.add_constant("const_12602")
        self.add_constant("const_02602")
        self.add_constant("const_22602")

        self.set_constant("features.init_block.conv1.bn.running_var", torch.rand(1, 1))
        self.set_constant("features.init_block.conv1.bn.running_mean", torch.rand(1, 1))
        self.set_constant("const_12602", torch.rand(1, 1))
        self.set_constant("const_02602", torch.rand(1, 1))
        self.set_constant("const_22602", torch.rand(1, 1))

        for param in self.get_parameters():
            self.set_parameter(param.get_name(), torch.rand(size = param.shape.get_pytorch_shape()))

    def forward(self, act):

        conv2d_1632 = forge.op.Conv2d("", act, self.get_parameter("features.init_block.conv1.conv.weight"), stride=[2, 2], padding=[1, 1, 1, 1], dilation=1, groups=1, channel_last=0)
        add_1635 = forge.op.Add("", self.get_constant("features.init_block.conv1.bn.running_var"), self.get_constant("const_12602"))
        sqrt_1636 = forge.op.Sqrt("", add_1635)
        reciprocal_1637 = forge.op.Reciprocal("", sqrt_1636)
        multiply_1638 = forge.op.Multiply("", self.get_constant("const_02602"), reciprocal_1637)
        multiply_1639 = forge.op.Multiply("", multiply_1638, self.get_parameter("features.init_block.conv1.bn.weight"))
        reshape_1640 = forge.op.Reshape("", multiply_1639, shape=(64, 1, 1))
        multiply_1641 = forge.op.Multiply("", conv2d_1632, reshape_1640)
        multiply_1643 = forge.op.Multiply("", self.get_constant("features.init_block.conv1.bn.running_mean"), self.get_constant("const_22602"))
        multiply_1644 = forge.op.Multiply("", multiply_1643, multiply_1639)
        add_1645 = forge.op.Add("", multiply_1644, self.get_parameter("features.init_block.conv1.bn.bias"))
        reshape_1646 = forge.op.Reshape("", add_1645, shape=(64, 1, 1))
        add_1647 = forge.op.Add("", multiply_1641, reshape_1646)
        relu_1648 = forge.op.Relu("", add_1647)

        conv2d_1649 = forge.op.Conv2d("", relu_1648, self.get_parameter("features.init_block.conv2.conv.weight"), self.get_parameter("bias1"), stride=[2, 2], padding=[1, 1, 1, 1], dilation=1, groups=1, channel_last=0)
        relu_1665 = forge.op.Relu("", conv2d_1649)
        conv2d_1666 = forge.op.Conv2d("", relu_1665, self.get_parameter("features.init_block.subblocks.block1.body.conv1.conv.weight"), self.get_parameter("bias2"), stride=[1, 1], padding=[0, 0, 0, 0], dilation=1, groups=1, channel_last=0)
        relu_1682 = forge.op.Relu("", conv2d_1666)
        conv2d_1683 = forge.op.Conv2d("", relu_1682, self.get_parameter("features.init_block.subblocks.block1.body.conv2.conv.weight"), self.get_parameter("bias3"), stride=[1, 1], padding=[1, 1, 1, 1], dilation=1, groups=1, channel_last=0)
        relu_1699 = forge.op.Relu("", conv2d_1683)
        conv2d_1700 = forge.op.Conv2d("", relu_1699, self.get_parameter("features.init_block.subblocks.block1.body.conv3.conv.weight"), self.get_parameter("bias4"), stride=[1, 1], padding=[0, 0, 0, 0], dilation=1, groups=1, channel_last=0)

        # Left side fork
        conv2d_1716 = forge.op.Conv2d("", relu_1665, self.get_parameter("features.init_block.subblocks.block1.identity_conv.conv.weight"), self.get_parameter("bias5"), stride=[1, 1], padding=[0, 0, 0, 0], dilation=1, groups=1, channel_last=0)

        # Join
        add_1732 = forge.op.Add("", conv2d_1700, conv2d_1716)
        relu_1733 = forge.op.Relu("", add_1732)

        conv2d_1734 = forge.op.Conv2d("", relu_1733, self.get_parameter("features.init_block.subblocks.block2.body.conv1.conv.weight"), stride=[1, 1], padding=[0, 0, 0, 0], dilation=1, groups=1, channel_last=0)

        return conv2d_1734

    @staticmethod
    def add_overrides():
        forge.config.override_op_size("conv2d_14.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (7, 1))
        forge.config.override_t_stream_shape("conv2d_14.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (7, 1))
        forge.config.override_u_kt("conv2d_14.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", 28)

        # Fork node
        forge.config.override_op_size("conv2d_14.dc.matmul.11", (2, 2))
        forge.config.override_t_stream_shape("conv2d_14.dc.matmul.11", (7, 1))
        forge.config.override_u_kt("conv2d_14.dc.matmul.11", 18)

        # Short path
        forge.config.override_op_size("conv2d_21.dc.matmul.8", (2, 4))
        forge.config.override_t_stream_shape("conv2d_21.dc.matmul.8", (7, 1))
        forge.config.override_u_kt("conv2d_21.dc.matmul.8", 1)

        # Long path
        forge.config.override_op_size("conv2d_16.dc.matmul.8", (2, 1))
        forge.config.override_t_stream_shape("conv2d_16.dc.matmul.8", (7, 1))
        forge.config.override_u_kt("conv2d_16.dc.matmul.8", 1)
        forge.config.override_op_size("conv2d_18.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (6, 2))
        forge.config.override_t_stream_shape("conv2d_18.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (1, 1))
        forge.config.override_u_kt("conv2d_18.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", 7)
        forge.config.override_op_size("conv2d_18.dc.matmul.11", (1, 1))
        forge.config.override_t_stream_shape("conv2d_18.dc.matmul.11", (1, 1))
        forge.config.override_u_kt("conv2d_18.dc.matmul.11", 1)
        forge.config.override_op_size("conv2d_20.dc.matmul.8", (2, 4))
        forge.config.override_t_stream_shape("conv2d_20.dc.matmul.8", (7, 1))
        forge.config.override_u_kt("conv2d_20.dc.matmul.8", 1)

        # Join
        forge.config.override_op_size("add_22", (2, 1))
        forge.config.override_t_stream_shape("add_22", (7, 1))

def test_fork_join_hrnet(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip("Skipping training test")

    if test_device.arch == forge.BackendDevice.Grayskull:
        pytest.skip("There is not enough L1 memory on Grayskull to fit some of these ops.")

    channels = 3
    height = 224
    width = 224

    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = DataFormat.Float16_b

    import os
    os.environ["FORGE_RIBBON2"] = "1"

    HRNetForkJoin.add_overrides()

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    verify_module(HRNetForkJoin("test_fork_join_hrnet"), [(1, channels, height, width)],
                  VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, pcc=pcc, relative_atol=relative_atol))

class ForkJoinExpandOutputBuffer(forge.ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.weights0 = forge.Parameter(1, 64, 128, requires_grad=False)

    def forward(self, act1):
        fork = forge.op.Matmul("matmul", act1, self.weights0)
        left = forge.op.Exp("exp", fork)
        right = forge.op.Buffer("buffer", fork)
        join = forge.op.Add("add", left, right)
        return join

# Test implementation of Backend constrains for buf_size_mb.
def test_fork_join_expand_output_buffer_constraints(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip("Skipping training test")
        
    forge.config.override_op_size("matmul", (2, 1))
    forge.config.override_op_size("exp", (2, 4))
    forge.config.override_t_stream_shape("matmul", (10, 1))
    forge.config.override_t_stream_shape("exp", (1, 1))

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    verify_module(ForkJoinExpandOutputBuffer("test_fork_join_expand_output_buffer_constraints"), [(1, 1, 6400, 64)],
                  VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, pcc=pcc, relative_atol=relative_atol))
