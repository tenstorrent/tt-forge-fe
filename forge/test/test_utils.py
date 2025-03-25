# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import torch.nn as nn

import tensorflow as tf

import forge
import forge.config
from forge.tensor import to_forge_tensors, to_pt_tensors
from forge.verify.value_checkers import AutomaticValueChecker


@pytest.mark.xfail(reason="This test is expected to fail")
def test_value_check_memory():
    fw_out = torch.rand(128256, 2048)
    co_out = torch.rand(128256, 2048)

    value_checker = AutomaticValueChecker(pcc=0.99, rtol=1e-05, atol=1e-08, dissimilarity_threshold=1e-03)
    value_checker.check(fw_out, co_out)
