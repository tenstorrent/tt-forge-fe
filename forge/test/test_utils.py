# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from math import isinf, isnan
import os
import pytest
import torch
import torch.nn as nn

import numpy as np

import forge
import forge.config
from forge.tensor import to_forge_tensors, to_pt_tensors
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.compare import calculate_pcc, calculate_pcc_unoptimized, calculate_atol


@pytest.mark.xfail(reason="This test is expected to fail")
def test_value_check_memory():
    fw_out = torch.rand(128256, 2048)
    co_out = torch.rand(128256, 2048)

    value_checker = AutomaticValueChecker(pcc=0.99, rtol=1e-05, atol=1e-08, dissimilarity_threshold=1e-03)
    value_checker.check(fw_out, co_out)


def test_pcc_memory():
    fw_out = torch.rand(128256, 2048)
    co_out = torch.rand(128256, 2048)

    pcc = calculate_pcc(fw_out, co_out)


def test_pcc_calculation():
    fw_out = torch.rand(1, 10000000)
    co_out = fw_out.clone()

    pcc = calculate_pcc(fw_out, co_out)
    golden_pcc = np.min(np.corrcoef(fw_out.numpy().flatten(), co_out.numpy().flatten())).item()

    assert torch.allclose(torch.tensor(pcc), torch.tensor(golden_pcc))


def test_atol_memory():
    fw_out = torch.rand(128256, 2048)
    co_out = torch.rand(128256, 2048)

    pcc = calculate_atol(fw_out, co_out)


def test_atol_calculation():
    expected = torch.tensor([1.0, torch.nan, 3.0])
    actual = torch.tensor([1.0, 2.0, 3.0])

    assert isinf(calculate_atol(expected, actual))

    actual = torch.tensor([1.0, torch.nan, 3.0])
    assert calculate_atol(expected, actual) == 0

    actual = torch.tensor([1.0, torch.inf, 3.0])
    assert isinf(calculate_atol(expected, actual))

    expected = torch.tensor([1.0, torch.inf, 3.0])
    actual = torch.tensor([1.0, torch.inf, 3.0])
    assert calculate_atol(expected, actual) == 0

    actual = torch.tensor([1.0, torch.nan, 3.0])
    assert isinf(calculate_atol(expected, actual))

    actual = torch.tensor([1.0, -torch.inf, 3.0])
    assert isinf(calculate_atol(expected, actual))

    expected = torch.tensor([True, False, True])
    actual = torch.tensor([True, False, True])

    actual = torch.tensor([True, False, False])
    assert calculate_atol(expected, actual) == 1
