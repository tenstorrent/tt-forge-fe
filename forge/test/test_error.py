# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test various error conditions, make sure they are caught and reported
"""

import pytest
import torch

import forge
from forge import CPUDevice, TTDevice, PyTorchModule, set_device_pipeline, ForgeModule, Tensor
from .test_user import ForgeTestModule, _safe_read

class BudaMatmul(ForgeModule):
    """
    Simple buda module for basic testing
    """

    def __init__(self, name):
        super().__init__(name)
        self.weights = forge.Parameter(32, 32, requires_grad=True)

    def forward(self, act):
        return forge.op.Matmul("matmul", act, self.weights)

def test_forge_on_cpu_device():

    lin = ForgeModule("lin")
    dev = CPUDevice("gs0")

    with pytest.raises(RuntimeError):
        dev.place_module(lin) # only pytorch modules on cpu devices

def test_invalid_modules():
    dev0 = TTDevice("gs0", devtype=forge.BackendType.Golden)
    dev1 = CPUDevice("gs1")

    with pytest.raises(RuntimeError):
        dev0.place_module("goo")

    with pytest.raises(RuntimeError):
        dev1.place_module([None, torch.nn.Linear(32, 32)])

def test_non_pytorch_module():

    with pytest.raises(RuntimeError):
        m = PyTorchModule("mod", lambda x: x + 1)


def test_invalid_dev_pipeline():
    with pytest.raises(RuntimeError):
        set_device_pipeline([0, 1])  # expects devices

def test_multiple_modules_with_same_name():
    m0 = PyTorchModule("linear", torch.nn.Linear(32, 32, bias=False).half())
    m1 = PyTorchModule("linear", torch.nn.Linear(32, 32, bias=False).half())
    with pytest.raises(RuntimeError):
        cpu = CPUDevice("cpu0")
        cpu.place_module(m0)
        cpu.place_module(m1)


def test_different_batch_inputs():
    dev0 = TTDevice("gs0", devtype=forge.BackendType.Golden)
    with pytest.raises(Exception):
        dev0.place_module(ForgeTestModule("placed"))

        # Compile & initialize the pipeline for inference, with given shapes
        output_q = forge.initialize_pipeline(training=False, sample_inputs=(torch.rand(4, 32, 32), torch.rand(4, 32, 32)))

        input1 = torch.rand(4, 32, 32)
        input2 = torch.rand(4, 32, 32)
        dev0.push_to_inputs((input1, input2))
        forge.run_forward(input_count=1)
        print(_safe_read(output_q))
        
        input1 = torch.rand(2, 32, 32)
        input2 = torch.rand(2, 32, 32)
        dev0.push_to_inputs((input1, input2))
        forge.run_forward(input_count=1)
        print(_safe_read(output_q))
