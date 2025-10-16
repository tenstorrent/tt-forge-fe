# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from torch import nn
from forge import forge
from forge.verify.verify import verify


class MatmulReshapeAddVIT(nn.Module):
    """
    Reproduces the exact VIT pattern that triggers the reshape issue.

    Pattern from VIT error:
    matmul: A x B -> [1576, 768] (1,210,368 elements)
    reshape: [1576, 768] -> [8, 197, 768]
    add: [8, 197, 768] + bias[768] -> [8, 197, 768]

    The bias [768] broadcasts to [8, 197, 768] but cannot be reshaped to [1576, 768].
    """

    def __init__(self):
        super().__init__()

    def forward(self, a, b, bias):
        # a: [1576, 768]
        # b: [768, 768]
        # bias: [768]
        matmul_out = torch.matmul(a, b)  # [1576, 768]
        reshaped = matmul_out.reshape(8, 197, 768)  # [8, 197, 768]
        result = reshaped + bias  # [8, 197, 768] (bias broadcasts from [768])
        return result


@pytest.mark.push
def test_matmul_reshape_add_vit_pattern():
    """
    Test the exact VIT pattern that was causing the error.
    This reproduces: loc("Add_16"): error: 'ttir.reshape' op Input tensor
    number of elements 768 and output tensor number of elements 1210368 must be the same
    """
    model = MatmulReshapeAddVIT()

    # Create inputs with exact VIT shapes
    a = torch.randn(1576, 768)
    b = torch.randn(768, 768)
    bias = torch.randn(768)

    inputs = [a, b, bias]

    compiled_model = forge.compile(model, sample_inputs=inputs)
    verify(inputs, model, compiled_model)


if __name__ == "__main__":
    test_matmul_reshape_add_vit_pattern()
