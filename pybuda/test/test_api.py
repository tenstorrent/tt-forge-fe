import os
import pytest
import torch
import torch.nn as nn

import tensorflow as tf

import pybuda
import pybuda.config

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

    compiled_model = pybuda.compile(model, sample_inputs=[torch.rand(shape), torch.rand(shape)])

    output = compiled_model(*inputs)

    print(f"golden: {golden}")
    print(f"output: {output}")
    if not torch.allclose(output[0], golden, rtol=1e-1):
        raise ValueError("Output does not match the golden output")

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

    compiled_model = pybuda.compile(model, sample_inputs=[torch.rand(shape), torch.rand(shape)])

    output = compiled_model(*inputs)

    print(f"golden: {golden}")
    print(f"output: {output}")
    if not torch.allclose(output[0], golden, rtol=1e-1):
        raise ValueError("Output does not match the golden output")
