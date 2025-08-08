# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from third_party.tt_forge_models.qwen_2.token_classification.pytorch import ModelLoader

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.models_utils import TextModelWrapper


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["Qwen/Qwen2-7B"])
def test_qwen2_token_classification(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.QWENV2,
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )
    pytest.xfail(reason="Requires multi-chip support")

    # Load model and inputs
    loader = ModelLoader()
    model = loader.load_model()
    model.config.use_cache = False
    framework_model = TextModelWrapper(model=model)
    input_dict = loader.load_inputs()
    inputs = [input_dict["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
