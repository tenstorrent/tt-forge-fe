# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import forge
import pytest

from test.models.pytorch.timeseries.nbeats.utils.dataset import get_electricity_dataset_input
from test.models.pytorch.timeseries.nbeats.utils.model import (
    NBeatsWithGenericBasis,
    NBeatsWithTrendBasis,
    NBeatsWithSeasonalityBasis,
)
from test.models.utils import build_module_name, Framework
from forge.verify.verify import verify


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_nbeats_with_seasonality_basis(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="nbeats", variant="seasionality_basis")

    record_forge_property("module_name", module_name)

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
    compiled_model = forge.compile(pytorch_model, sample_inputs=[x, x_mask], module_name=module_name)
    inputs = [x, x_mask]

    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_nbeats_with_generic_basis(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="nbeats", variant="generic_basis")

    record_forge_property("module_name", module_name)

    x, x_mask = get_electricity_dataset_input()

    pytorch_model = NBeatsWithGenericBasis(input_size=72, output_size=24, stacks=30, layers=4, layer_size=512)
    pytorch_model.eval()

    compiled_model = forge.compile(pytorch_model, sample_inputs=[x, x_mask], module_name=module_name)
    inputs = [x, x_mask]

    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_nbeats_with_trend_basis(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="nbeats", variant="trend_basis")

    record_forge_property("module_name", module_name)

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

    compiled_model = forge.compile(pytorch_model, sample_inputs=[x, x_mask], module_name=module_name)
    inputs = [x, x_mask]

    verify(inputs, framework_model, compiled_model)
