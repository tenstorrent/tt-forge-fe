# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from math import isinf
import torch

import numpy as np

from forge.verify.compare import calculate_pcc, calculate_or_estimate_pcc, calculate_atol
import pytest

import forge._C.verif as verif


@pytest.mark.push
def test_pcc_estimation():
    fw_out = torch.rand(1, 5 * 1000000)
    co_out = fw_out.clone()

    # Estimate pcc by splitting the tensors into chunks.
    pcc = calculate_or_estimate_pcc(fw_out, co_out, tensor_size_threshold=1, chunk_size=1000000)
    golden_pcc = np.min(np.corrcoef(fw_out.numpy().flatten(), co_out.numpy().flatten()))

    assert torch.allclose(torch.tensor(pcc, dtype=torch.double), torch.tensor(golden_pcc, dtype=torch.double))

    # Estimate pcc by splitting the tensors into chunks. Simulate not completely equal tensors.
    fw_out = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float32)
    co_out = torch.tensor(
        [[1.1, 1.9, 3.1], [3.94, 4.98, 6.05], [6.9, 8.1, 9.1], [10.1, 11.02, 11.97]], dtype=torch.float32
    )

    pcc = calculate_pcc(fw_out, co_out)
    estimated_pcc = calculate_or_estimate_pcc(fw_out, co_out, tensor_size_threshold=1, chunk_size=3)

    assert torch.allclose(torch.tensor(pcc), torch.tensor(estimated_pcc), atol=1e-2)

    # Check scenario with different number of NaNs in the tensors. The PCC should be 0.
    fw_out = torch.tensor([[1, 2, 3, torch.nan, 5, 6, torch.nan, 8, 9, 10]])
    co_out = torch.tensor([[1.1, 1.9, 3.1, 3.94, 4.98, 6.05, 6.9, 8.1, 9.1, 10.1]])

    pcc = calculate_pcc(fw_out, co_out)

    assert torch.allclose(torch.tensor(pcc), torch.tensor(0.0))

    # Check scenario with the same number of NaNs in the tensors.
    fw_out = torch.tensor([[1, 2, 3, torch.nan, 5, 6, torch.nan, 8, 9, 10]])
    co_out = torch.tensor([[1.1, 1.9, 3.1, torch.nan, 4.98, 6.05, torch.nan, 8.1, 9.1, 10.1]])

    pcc = calculate_pcc(fw_out, co_out)

    assert pcc > 0.99

    # Check scenario with NaNs not matching (one has nan on index 3, the other on index 2).
    fw_out = torch.tensor([[1, 2, 3, torch.nan, 5, 6, torch.nan, 8, 9, 10]])
    co_out = torch.tensor([[1.1, 1.9, torch.nan, 3.94, 4.98, 6.05, torch.nan, 8.1, 9.1, 10.1]])

    pcc = calculate_pcc(fw_out, co_out)

    assert torch.allclose(torch.tensor(pcc), torch.tensor(0.0))

    # Check scenario with Nans/Infs not matching.
    fw_out = torch.tensor([[1, 2, 3, torch.nan, 5, 6, torch.nan, 8, 9, 10]])
    co_out = torch.tensor([[1.1, 1.9, 3.1, torch.inf, 4.98, 6.05, torch.nan, 8.1, 9.1, 10.1]])

    pcc = calculate_pcc(fw_out, co_out)
    assert torch.allclose(torch.tensor(pcc), torch.tensor(0.0))


@pytest.mark.push
def test_pcc_kernel():
    fw_out = torch.rand(1, 5 * 1000000)
    co_out = fw_out.clone()

    pcc = verif.calculate_tensor_pcc(fw_out, co_out)
    golden_pcc = np.min(np.corrcoef(fw_out.numpy().flatten(), co_out.numpy().flatten()))

    assert torch.allclose(torch.tensor(pcc, dtype=torch.double), torch.tensor(golden_pcc, dtype=torch.double))


@pytest.mark.push
def test_atol_calculation():
    # For all of the cases where an entry in the tensor is expected to be NaN/+Inf/-Inf,
    # and it is not, the atol should be inf. (explanation being that we are infinitely wrong in any case)

    # Check scenario with NaNs mismatch. Calculated atol should be inf.
    expected = torch.tensor([1.0, torch.nan, 3.0])
    actual = torch.tensor([1.0, 2.0, 3.0])

    assert isinf(calculate_atol(expected, actual))

    # Check scenario with NaNs matching. Calculated atol should be 0.
    actual = torch.tensor([1.0, torch.nan, 3.0])
    assert calculate_atol(expected, actual) == 0

    # Inf instead of NaN. Calculated atol should be inf.
    actual = torch.tensor([1.0, torch.inf, 3.0])
    assert isinf(calculate_atol(expected, actual))

    # Check scenario with Infs matching. Calculated atol should be 0.
    expected = torch.tensor([1.0, torch.inf, 3.0])
    actual = torch.tensor([1.0, torch.inf, 3.0])
    assert calculate_atol(expected, actual) == 0

    # NaN instead of inf. Calculated atol should be inf.
    actual = torch.tensor([1.0, torch.nan, 3.0])
    assert isinf(calculate_atol(expected, actual))

    # -Inf instead of +inf. Calculated atol should be inf.
    actual = torch.tensor([1.0, -torch.inf, 3.0])
    assert isinf(torch.tensor(calculate_atol(expected, actual)))

    # Check equal booleans scenario. Even though it doesn't make sense to compare bools with tolerances,
    # we return 0.0 or 1.0 depending if all of the elements are equal or not.
    expected = torch.tensor([True, False, True])
    actual = torch.tensor([True, False, True])
    assert calculate_atol(expected, actual) == 0

    # Check non-equal booleans scenario.
    actual = torch.tensor([True, False, False])
    assert calculate_atol(expected, actual) == 1
