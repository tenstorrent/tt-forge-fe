# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch

import forge
from forge import (
    Tensor,
    CompilerConfig,
    CompileDepth,
    VerifyConfig,
)
from .common import compile


def test_max_input_grid_fork():
    @compile(
        verify_cfg=VerifyConfig(run_golden=False)
    )
    def max_input_grid_fork(a, b):
        b = forge.op.Repeat("repeat", b, [1, 1, 3, 6])
        return forge.op.Add("add0", a, b)

    forge.config.override_op_size("add0", (3, 6))

    a = Tensor.create_from_torch(torch.rand((1, 1, 96, 192)))
    b = Tensor.create_from_torch(torch.rand((1, 1, 32, 32)))
    try:
        max_input_grid_fork(a, b)
    except RuntimeError as e:
        assert str(e) == "Could not satisfy all constraints for edge: input_0_add0 -> add0"


def test_max_input_op_fork():
    max_forks = 16
    @compile(
        verify_cfg=VerifyConfig(run_golden=False)
    )
    def max_input_op_fork(a, b):
        for i in range(max_forks + 1):
            b = forge.op.Add(f"add{i}", a, b)
        return b

    a = Tensor.create_from_torch(torch.rand((1, 1, 96, 192)))
    b = Tensor.create_from_torch(torch.rand((1, 1, 96, 192)))
    max_input_op_fork(a, b)


def test_max_prologue_op_fork():
    max_forks = 16
    @compile(
        verify_cfg=VerifyConfig(run_golden=False)
    )
    def max_prologue_op_fork(a, const=None):
        return forge.op.Multiply(f"op0", a, const)

    rt = 4
    ct = 6
    assert rt * ct > max_forks
    forge.config.override_op_size("op0", (rt, ct))
    a = Tensor.create_from_torch(torch.rand((1, 1, rt*32, ct*32)))
    c = Tensor.create_from_torch(torch.rand((1, 1, 1, 1)), constant=True)
    max_prologue_op_fork(a, const=c)


def test_max_output_op_fork():
    max_forks = 8
    @compile(
        verify_cfg=VerifyConfig(run_golden=False)
    )
    def max_output_op_fork(a, b):
        outputs = []
        b = forge.op.Add(f"add_fork", a, b)
        for i in range(max_forks + 1):
            outputs.append(forge.op.Exp(f"exp{i}", b))
        return outputs

    for i in range(max_forks + 1):
        forge.config.set_epoch_break(f"exp{i}")

    a = Tensor.create_from_torch(torch.rand((1, 1, 96, 192)))
    b = Tensor.create_from_torch(torch.rand((1, 1, 96, 192)))
    max_output_op_fork(a, b)


def test_max_fork_streams():
    @compile(
        verify_cfg=VerifyConfig(run_golden=False)
    )
    def max_fork_streams(a, b):
        c = forge.op.Add("add0", a, b)
        d = forge.op.Add("add1", a, b)
        e = forge.op.Add("add2", c, d)
        return e

    forge.config.override_op_size("add0", (1, 1))
    forge.config.override_op_size("add1", (2, 4))
    forge.config.override_op_size("add2", (4, 8))

    a = Tensor.create_from_torch(torch.rand((1, 1, 128, 256)))
    b = Tensor.create_from_torch(torch.rand((1, 1, 128, 256)))
    try:
        max_fork_streams(a, b)
    except RuntimeError as e:
        assert str(e) == "Could not satisfy all constraints for edge: add0 -> add2"


def test_stream_stacking_rotate():
    forge.config.set_configuration_options(balancer_policy="MaximizeTMinimizeGrid")

    @compile(
        verify_cfg=VerifyConfig(run_golden=False, run_net2pipe=True)
    )
    def stream_stacking_rotate(a, b, c):
        x = forge.op.Matmul("mm0", a, b)

        c = forge.op.HSlice("h0", c, 12)
        x = forge.op.Transpose("t1", x, 2, 3)
        x = forge.op.VSlice("v1", x, 12)
        r = forge.op.Matmul("mm1", c, x)
        return r

    a = Tensor.create_from_torch(torch.rand((1, 1, 384, 768)))
    b = Tensor.create_from_torch(torch.rand((1, 1, 768, 768)))
    c = Tensor.create_from_torch(torch.rand((1, 1, 768, 768)))
    stream_stacking_rotate(a, b, c)


def test_stream_stacking_transpose():
    forge.config.set_configuration_options(balancer_policy="MaximizeTMinimizeGrid")

    @compile(
        verify_cfg=VerifyConfig(run_golden=False, run_net2pipe=True)
    )
    def stream_stacking_transpose(a, b, c):
        b = forge.op.Matmul("mm0", b, c)
        b = forge.op.Transpose("transpose0", b, 2, 3)
        b = forge.op.VStack("vstack0", b, 512)
        r = forge.op.Matmul("mm1", a, b)
        return r

    a = Tensor.create_from_torch(torch.rand((1, 1, 128, 32*512)))
    b = Tensor.create_from_torch(torch.rand((1, 512, 64, 32)))
    c = Tensor.create_from_torch(torch.rand((1, 512, 32, 32)))
    stream_stacking_transpose(a, b, c)


def test_r_stream_mm_rhs():
    forge.config.set_configuration_options(balancer_policy="MaximizeTMinimizeGrid")
    forge.config._get_global_compiler_config().insert_queues = [("exp0", "mm1", 1)]

    @compile(
        verify_cfg=VerifyConfig(run_golden=False, run_net2pipe=True)
    )
    def r_stream_mm_rhs(a, b):
        b = forge.op.Exp("exp0", b)
        r = forge.op.Matmul("mm1", a, b)
        return r

    a = Tensor.create_from_torch(torch.rand((1, 1, 128, 128)))
    b = Tensor.create_from_torch(torch.rand((1, 1, 128, 128)))
    r_stream_mm_rhs(a, b)


def test_queue_fork_streams():
    @compile()
    def queue_fork_streams(a, b, w=None):
        c = forge.op.Add("add0", a, b)
        d = forge.op.Matmul("mm0", c, w)
        return d

    grid = (7, 8)
    forge.config.override_op_size("add0", grid)
    forge.config.set_epoch_break("mm0")
    forge.config.override_op_size("mm0", (1, 1))

    a = Tensor.create_from_torch(torch.rand((1, 1, 32*grid[0], 32*grid[1])))
    b = Tensor.create_from_torch(torch.rand((1, 1, 32*grid[0], 32*grid[1])))
    w = Tensor.create_from_torch(torch.rand((1, 1, 256, 32*grid[0])))
    try:
        queue_fork_streams(a, b, w=w)
    except RuntimeError as e:
        assert str(e).startswith("Could not satisfy all constraints for edge")


def test_aggregate_queue_fork_streams():
    grid = (8, 1)
    max_queue_streams = 40
    num_adds = (max_queue_streams // (grid[0] * grid[1]))

    @compile()
    def aggregate_queue_fork_streams(a, b):
        outs = []
        for i in range(num_adds):
            a = forge.op.Add(f"add{i}", a, b)
            outs.append(a)
        return forge.op.Concatenate("concat0", *outs, axis=-1)

    for i in range(num_adds):
        forge.config.override_op_size(f"add{i}", grid)

    forge.config.set_epoch_break("concat0.dc.concatenate.0")
    forge.config.override_op_size("concat0.dc.concatenate.0", (1, 1))

    a = Tensor.create_from_torch(torch.rand((1, 1, 32*grid[0], 32*grid[1])))
    b = Tensor.create_from_torch(torch.rand((1, 1, 32*grid[0], 32*grid[1])))
    try:
        aggregate_queue_fork_streams(a, b)
    except RuntimeError as e:
        assert str(e).startswith("Fatal balancer error: Could not reconcile constraints")


def test_epoch_to_epoch_disjoint():
    grid = (8, 8)
    compiler_cfg=CompilerConfig(enable_training=True)
    compiler_cfg.balancer_op_override("mm0", "grid_shape", (1, 1))
    compiler_cfg.balancer_op_override("bw_in0_mul0_multiply_0", "grid_shape", grid)

    @compile(compiler_cfg=compiler_cfg)
    def epoch_to_epoch_disjoint(a, b, w=None):
        c = forge.op.Matmul("mm0", a, w)
        return forge.op.Multiply("mul0", b, c)

    a = Tensor.create_from_torch(torch.rand((1, 1, 32*grid[0], 32*grid[1]), requires_grad=True))
    w = Tensor.create_from_torch(torch.rand((1, 1, 32*grid[1], 32*grid[0]), requires_grad=True))
    b = Tensor.create_from_torch(torch.rand((1, 1, 32*grid[0], 32*grid[1]), requires_grad=True))
    epoch_to_epoch_disjoint(a, b, w=w)
