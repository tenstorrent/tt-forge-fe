# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from third_party.tt_forge_models.deepcogito.pytorch import ModelLoader, ModelVariant

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.text.deepcogito.model_utils.model import CogitoWrapper


@pytest.mark.out_of_memory
@pytest.mark.xfail
@pytest.mark.nightly
@pytest.mark.parametrize("variant", [ModelVariant.V1_PREVIEW_LLAMA_3B])
def test_cogito_generation(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.COGITO,
        variant=variant,
        task=Task.TEXT_GENERATION,
        source=Source.HUGGINGFACE,
    )
    pytest.xfail(reason="Requires multi-chip support")

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    framework_model = CogitoWrapper(model)
    framework_model.eval()
    inputs_dict = loader.load_inputs()
    sample_inputs = [inputs_dict["input_ids"]]

    # Compile with Forge
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=sample_inputs,
        module_name=module_name,
    )

    # Run verification
    verify(sample_inputs, framework_model, compiled_model)
