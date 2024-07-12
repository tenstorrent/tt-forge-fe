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
    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 32)]

    compiled_model = pybuda.compile(model, sample_inputs=[torch.rand(1, 32, 32), torch.rand(1, 32, 32)])

    # TODO: Run inference on the compiled model, in the following way:
    # compiled_model(*inputs)

def test_tf():
    class TFAdd(tf.keras.Model):
        def __init__(self):
            super().__init__()

        def call(self, x1, x2):
            return x1 + x2

    model = TFAdd()
    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 32)]
    pybuda.compile(model, sample_inputs=[torch.rand(1, 32, 32), torch.rand(1, 32, 32)])

    # TODO: Run inference on the compiled model, in the following way:
    # compiled_model(*inputs)
