# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import forge

from forge.verify.verify import verify


def masked_scatter_simulation(input_tensor, mask, source):
    """
    Simulate the behavior of masked_scatter as an out-of-place operation.

    Args:
        input_tensor (torch.Tensor): The target tensor.
        mask (torch.Tensor): A boolean tensor specifying positions to replace.
        source (torch.Tensor): The source tensor providing values to insert.

    Returns:
        torch.Tensor: A new tensor with values from `source` scattered into `input_tensor` based on `mask`.
    """
    # Ensure input tensors are compatible
    # if mask.dtype != torch.bool:
    #     raise ValueError("mask must be a boolean tensor.")
    # if source.numel() != mask.sum():
    #     raise ValueError("The number of elements in `source` must match the number of `True` values in `mask`.")

    # Clone the input tensor to avoid in-place modifications
    result = input_tensor.clone()

    # Use index_put_ with the mask to scatter the source values
    result[mask] = source

    return result


@pytest.mark.parametrize(
    "input_tensor, mask, source_tensor",
    [
        (torch.tensor([1, 2, 3, 4]), torch.tensor([True, False, True, False]), torch.tensor([10, 20])),
        (torch.tensor([[1, 2], [3, 4]]), torch.tensor([[True, False], [False, True]]), torch.tensor([10, 20])),
    ],
)
def test_mark_scatter_alternative(input_tensor, mask, source_tensor):
    # Apply the simulated masked_scatter operation
    result = masked_scatter_simulation(input_tensor, mask, source_tensor)

    # Verify the result matches the expected output
    assert torch.allclose(result, input_tensor.masked_scatter(mask, source_tensor))

    print(f"\nInput Tensor: \n{input_tensor}\n")
    print(f"Mask: \n{mask}\n")
    print(f"Source Tensor: \n{source_tensor}\n")
    print(f"Result: \n{result}\n")


@pytest.mark.parametrize(
    "input_tensor, mask, source_tensor",
    [
        # (torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([True, False, True, False]), torch.tensor([10.0, 20.0])),
        (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[True, False], [False, True]]),
            torch.tensor([10.0, 20.0]),
        ),
    ],
)
def test_mark_scatter_alternative_module(input_tensor, mask, source_tensor):
    class MaskedScatterModule(torch.nn.Module):
        def forward(self, input_tensor, mask, source_tensor):
            return masked_scatter_simulation(input_tensor, mask, source_tensor)

    inputs = [input_tensor, mask, source_tensor]

    # Create an instance of the MaskedScatterModule
    framework_model = MaskedScatterModule()

    # Compile the model
    compiled_model = forge.compile(framework_model, inputs)

    # Verify the compiled model produces the expected output
    verify(inputs, framework_model, compiled_model)
