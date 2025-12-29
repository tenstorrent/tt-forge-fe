# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from third_party.tt_forge_models.roberta.masked_lm.pytorch import (
    ModelLoader as MaskedLMLoader,
)
from third_party.tt_forge_models.roberta.masked_lm.pytorch import (
    ModelVariant as MaskedLMVariant,
)
from third_party.tt_forge_models.roberta.pytorch import ModelLoader, ModelVariant

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from test.models.pytorch.text.roberta.model_utils.model_utils import RobertaWrapper


@pytest.mark.nightly
@pytest.mark.parametrize("variant", [MaskedLMVariant.XLM_BASE])
def test_roberta_masked_lm(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.ROBERTA,
        variant=variant.value,
        task=Task.NLP_MASKED_LM,
        source=Source.HUGGINGFACE,
    )

    # Use masked LM loader to load model and inputs
    loader = MaskedLMLoader(variant)
    framework_model = loader.load_model()
    framework_model = RobertaWrapper(framework_model)
    input_dict = loader.load_inputs()
    inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing using loader's decode
    print(loader.decode_output(co_out))


@pytest.mark.nightly
@pytest.mark.parametrize("variant", [ModelVariant.ROBERTA_BASE_SENTIMENT])
def test_roberta_sentiment_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.ROBERTA,
        variant=variant.value,
        task=Task.NLP_SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load model and inputs using the new loader
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model()
    input_tokens = loader.load_inputs()
    inputs = [input_tokens]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    _, co_out = verify(
        inputs, framework_model, compiled_model, verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.98))
    )

    # post processing using loader's decode method
    loader.decode_output(co_out)
