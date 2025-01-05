# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import forge
import pytest
import sys
import os

from test.models.pytorch.timeseries.nbeats.utils.dataset import get_electricity_dataset_input
from test.models.pytorch.timeseries.nbeats.utils.model import (
    NBeatsWithGenericBasis,
    NBeatsWithTrendBasis,
    NBeatsWithSeasonalityBasis,
)
import torch
from forge.verify.compare import compare_with_golden
from test.models.utils import build_module_name


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_nbeats_with_seasonality_basis(test_device):
    x, x_mask = get_electricity_dataset_input()

    pytorch_model = NBeatsWithSeasonalityBasis(
        input_size=72,
        output_size=24,
        num_of_harmonics=1,
        stacks=30,
        layers=4,
        layer_size=2048,
    )
    pytorch_model.eval()
    module_name = build_module_name(framework="pt", model="nbeats", variant=variant, task="seasionality_basis")
    compiled_model = forge.compile(pytorch_model, sample_inputs=[x, x_mask], module_name=module_name)
    inputs = [x, x_mask]
    co_out = compiled_model(*inputs)
    fw_out = pytorch_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_nbeats_with_generic_basis(test_device):
    x, x_mask = get_electricity_dataset_input()

    pytorch_model = NBeatsWithGenericBasis(input_size=72, output_size=24, stacks=30, layers=4, layer_size=512)
    pytorch_model.eval()

    module_name = build_module_name(framework="pt", model="nbeats", variant=variant, task="generic_basis")
    compiled_model = forge.compile(pytorch_model, sample_inputs=[x, x_mask], module_name=module_name)
    inputs = [x, x_mask]
    co_out = compiled_model(*inputs)
    fw_out = pytorch_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_nbeats_with_trend_basis(test_device):
    x, x_mask = get_electricity_dataset_input()

    pytorch_model = NBeatsWithTrendBasis(
        input_size=72,
        output_size=24,
        degree_of_polynomial=3,
        stacks=30,
        layers=4,
        layer_size=256,
    )
    pytorch_model.eval()

    module_name = build_module_name(framework="pt", model="nbeats", variant=variant, task="trend_basis")
    compiled_model = forge.compile(pytorch_model, sample_inputs=[x, x_mask], module_name=module_name)
    inputs = [x, x_mask]
    co_out = compiled_model(*inputs)
    fw_out = pytorch_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])
