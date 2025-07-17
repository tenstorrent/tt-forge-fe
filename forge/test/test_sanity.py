# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sanity test for forge add operation compilation and verification.
This test ensures that basic forge functionality works with element-wise addition.
"""
import pytest
import torch
import forge
from forge.verify.verify import verify


@pytest.mark.push
@pytest.mark.sanity
@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((4, 4), torch.float32),
        ((6, 7), torch.float32),
        ((2, 3, 4), torch.float32),
    ],
)
def test_eltwise_add(shape, dtype):
    """Test element-wise addition using forge compile and verify."""

    class AddModel(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    input1 = torch.randn(shape, dtype=dtype)
    input2 = torch.randn(shape, dtype=dtype)
    inputs = [input1, input2]

    model = AddModel()
    model.eval()

    compiled_model = forge.compile(model, sample_inputs=inputs)
    verify(inputs, model, compiled_model)
