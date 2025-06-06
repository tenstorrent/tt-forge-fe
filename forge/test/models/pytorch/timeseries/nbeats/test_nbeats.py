# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.timeseries.nbeats.model_utils.dataset import (
    get_electricity_dataset_input,
)
from test.models.pytorch.timeseries.nbeats.model_utils.model import (
    NBeatsWithGenericBasis,
    NBeatsWithSeasonalityBasis,
    NBeatsWithTrendBasis,
)


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", ["seasionality_basis"])
def test_nbeats_with_seasonality_basis(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.NBEATS,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

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
@pytest.mark.xfail
@pytest.mark.parametrize("variant", ["generic_basis"])
def test_nbeats_with_generic_basis(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.NBEATS,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    x, x_mask = get_electricity_dataset_input()

    framework_model = NBeatsWithGenericBasis(input_size=72, output_size=24, stacks=30, layers=4, layer_size=512)
    framework_model.eval()

    inputs = [x, x_mask]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", ["trend_basis"])
def test_nbeats_with_trend_basis(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.NBEATS,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

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
