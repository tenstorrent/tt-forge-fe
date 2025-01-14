# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.verify.verify import verify

from test.models.pytorch.timeseries.nbeats.utils.dataset import (
    get_electricity_dataset_input,
)
from test.models.pytorch.timeseries.nbeats.utils.model import (
    NBeatsWithGenericBasis,
    NBeatsWithSeasonalityBasis,
    NBeatsWithTrendBasis,
)
from test.models.utils import Framework, build_module_name


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["seasionality_basis"])
def test_nbeats_with_seasonality_basis(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="nbeats", variant=variant)

    # Record Forge Property
    record_forge_property("module_name", module_name)

    x, x_mask = get_electricity_dataset_input()

    framework_model = NBeatsWithSeasonalityBasis(
        input_size=72,
        output_size=24,
        num_of_harmonics=1,
        stacks=30,
        layers=4,
        layer_size=2048,
    )
    framework_model.eval()

    inputs = [x, x_mask]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["generic_basis"])
def test_nbeats_with_generic_basis(record_forge_property, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="nbeats", variant=variant)

    # Record Forge Property
    record_forge_property("module_name", module_name)

    x, x_mask = get_electricity_dataset_input()

    framework_model = NBeatsWithGenericBasis(input_size=72, output_size=24, stacks=30, layers=4, layer_size=512)
    framework_model.eval()

    inputs = [x, x_mask]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["trend_basis"])
def test_nbeats_with_trend_basis(record_forge_property, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="nbeats", variant=variant)

    # Record Forge Property
    record_forge_property("module_name", module_name)

    x, x_mask = get_electricity_dataset_input()

    framework_model = NBeatsWithTrendBasis(
        input_size=72,
        output_size=24,
        degree_of_polynomial=3,
        stacks=30,
        layers=4,
        layer_size=256,
    )
    framework_model.eval()

    inputs = [x, x_mask]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
