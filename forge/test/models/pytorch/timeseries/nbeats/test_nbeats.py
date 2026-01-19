# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from third_party.tt_forge_models.nbeats.pytorch import ModelLoader, ModelVariant

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", [ModelVariant.SEASONALITY_BASIS])
def test_nbeats_with_seasonality_basis(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.NBEATS,
        variant=variant.value,
        task=Task.TIME_SERIES_FORECASTING,
        source=Source.GITHUB,
    )

    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model()
    inputs = loader.load_inputs()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-onnx/issues/2928")
@pytest.mark.parametrize("variant", [ModelVariant.GENERIC_BASIS])
def test_nbeats_with_generic_basis(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.NBEATS,
        variant=variant.value,
        task=Task.TIME_SERIES_FORECASTING,
        source=Source.GITHUB,
    )

    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model()
    inputs = loader.load_inputs()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-onnx/issues/2928")
@pytest.mark.parametrize("variant", [ModelVariant.TREND_BASIS])
def test_nbeats_with_trend_basis(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.NBEATS,
        variant=variant.value,
        task=Task.TIME_SERIES_FORECASTING,
        source=Source.GITHUB,
    )

    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model()
    inputs = loader.load_inputs()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
