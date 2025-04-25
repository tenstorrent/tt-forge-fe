# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import torch.nn as nn

import tensorflow as tf

import forge
from forge.config import CompilerConfig, MLIRConfig
from forge.tensor import to_forge_tensors, to_pt_tensors
from forge.verify.value_checkers import AutomaticValueChecker


@pytest.mark.push
@pytest.mark.functional
def test_torch():
    class Add(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2):
            return torch.add(x1, x2)

    model = Add()
    shape = (1, 1024, 32)
    inputs = [torch.rand(shape), torch.rand(shape)]

    golden = model(*inputs)

    compiled_model = forge.compile(model, sample_inputs=[torch.rand(shape), torch.rand(shape)])

    output = compiled_model(*inputs)

    print(f"golden: {golden}")
    print(f"output: {output}")
    if not torch.allclose(output[0], golden, rtol=1e-1):
        raise ValueError("Output does not match the golden output")


@pytest.mark.push
@pytest.mark.functional
def test_tf():
    class TFAdd(tf.keras.Model):
        def __init__(self):
            super().__init__()

        def call(self, x1, x2):
            return x1 + x2

    model = TFAdd()
    shape = (1, 1024, 32)
    inputs = [torch.rand(shape), torch.rand(shape)]

    inputs_tf = [tf.convert_to_tensor(x) for x in inputs]
    golden = model(inputs_tf[0], inputs_tf[1])
    golden = torch.tensor(golden.numpy())

    compiled_model = forge.compile(model, sample_inputs=[torch.rand(shape), torch.rand(shape)])

    output = compiled_model(*inputs)

    print(f"golden: {golden}")
    print(f"output: {output}")
    if not torch.allclose(output[0], golden, rtol=1e-1):
        raise ValueError("Output does not match the golden output")


@pytest.mark.push
@pytest.mark.functional
def test_forge():
    class ForgeAdd(forge.ForgeModule):
        def __init__(self):
            super().__init__("ForgeTest")

        def forward(self, x, y):
            return forge.op.Add("", x, y)

    inputs = to_forge_tensors([torch.rand(1, 32, 32), torch.rand(1, 32, 32)])

    model = ForgeAdd()
    golden = model(*inputs)

    compiled_model = forge.compile(model, sample_inputs=inputs)

    # Issue #161 : currently, we expect inputs to be torch tensors
    inputs = to_pt_tensors(inputs)
    output = compiled_model(*inputs)

    print(f"golden: {golden}")
    print(f"output: {output}")

    if not torch.allclose(output[0], golden.to_pytorch(), rtol=1e-1):
        raise ValueError("Output does not match the golden output")


@pytest.mark.push
@pytest.mark.functional
def test_export_to_cpp():
    class Add(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2):
            return torch.add(x1, x2)

    model = Add()
    shape = (1, 1024, 32)
    inputs = [torch.rand(shape), torch.rand(shape)]

    compiled_model = forge.compile(model, sample_inputs=[torch.rand(shape), torch.rand(shape)])

    file_path = "generated_export_add.cpp"
    compiled_model.export_to_cpp(file_path)

    assert os.path.exists(file_path)
    with open(file_path, "r") as f:
        print(f.read())

    os.remove(file_path)


@pytest.mark.push
@pytest.mark.functional
# Sanity test for consteval pass in mlir (this test doesn't actually belong in this file, but
# at the moment it is the best fit)
def test_consteval_mlir():
    class ConstEvalParam(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.rand(32, 1024))
            self.const = torch.rand(1, 1)

        def forward(self, x):
            # This operation should be consteval'ed, since its arguments are constant and parameter
            w = torch.multiply(self.param, self.const)
            return torch.multiply(x, w)

    model = ConstEvalParam()
    shape = (32, 1024)
    inputs = [torch.rand(shape)]

    golden = model(*inputs)

    compiler_cfg = CompilerConfig()
    compiler_cfg.enable_consteval = False
    compiler_cfg.mlir_config = MLIRConfig().set_enable_consteval(True)

    compiled_model = forge.compile(model, sample_inputs=[torch.rand(shape)], compiler_cfg=compiler_cfg)

    output = compiled_model(*inputs)

    print(f"golden: {golden}")
    print(f"output: {output}")
    AutomaticValueChecker().check(
        output[0],
        golden,
    )
