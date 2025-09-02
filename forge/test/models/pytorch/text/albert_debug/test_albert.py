# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from loguru import logger

from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from third_party.tt_forge_models.albert.token_classification.pytorch import (
    ModelLoader as TokenClassificationLoader,
)
from third_party.tt_forge_models.albert.token_classification.pytorch import (
    ModelVariant as TokenClassificationVariant,
)

token_classification_params = [
    pytest.param(TokenClassificationVariant.XLARGE_V2, marks=[pytest.mark.xfail]),
]

import forge
from forge.verify.verify import verify


@pytest.mark.nightly
@pytest.mark.parametrize("variant", token_classification_params)
def test_albert_token_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.ALBERT,
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load Model and inputs
    loader = TokenClassificationLoader(variant=variant)
    framework_model = loader.load_model()
    framework_model.config.return_dict = False

    logger.info("framework_model={}", framework_model)

    input_dict = loader.load_inputs()
    inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # Post-processing
    predicted_tokens_classes = loader.decode_output(co_out, input_dict)

    print(f"Context: {loader.sample_text}")
    print(f"Answer: {predicted_tokens_classes}")
