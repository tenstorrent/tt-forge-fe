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
from forge.verify.config import AutomaticValueChecker, VerifyConfig
from forge.verify.verify import verify
from third_party.tt_forge_models.albert.masked_lm.pytorch import (
    ModelLoader as MaskedLMLoader,
)
from third_party.tt_forge_models.albert.masked_lm.pytorch import (
    ModelVariant as MaskedLMVariant,
)
from third_party.tt_forge_models.albert.question_answering.pytorch import (
    ModelLoader as QuestionAnsweringLoader,
)
from third_party.tt_forge_models.albert.question_answering.pytorch import (
    ModelVariant as QuestionAnsweringVariant,
)
from third_party.tt_forge_models.albert.sequence_classification.pytorch import (
    ModelLoader as SequenceClassificationLoader,
)
from third_party.tt_forge_models.albert.sequence_classification.pytorch import (
    ModelVariant as SequenceClassificationVariant,
)
from third_party.tt_forge_models.albert.token_classification.pytorch import (
    ModelLoader as TokenClassificationLoader,
)
from third_party.tt_forge_models.albert.token_classification.pytorch import (
    ModelVariant as TokenClassificationVariant,
)

masked_lm_params = [
    pytest.param(MaskedLMVariant.BASE_V1),
    pytest.param(MaskedLMVariant.LARGE_V1),
    pytest.param(MaskedLMVariant.XLARGE_V1),
    pytest.param(MaskedLMVariant.XXLARGE_V1),
    pytest.param(MaskedLMVariant.BASE_V2),
    pytest.param(MaskedLMVariant.LARGE_V2),
    pytest.param(MaskedLMVariant.XLARGE_V2),
    pytest.param(MaskedLMVariant.XXLARGE_V2),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", masked_lm_params)
def test_albert_masked_lm_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.ALBERT,
        variant=variant,
        task=Task.MASKED_LM,
        source=Source.HUGGINGFACE,
    )

    # Load Model and input
    loader = MaskedLMLoader(variant=variant)
    framework_model = loader.load_model()
    framework_model.config.return_dict = False
    input_dict = loader.load_inputs()
    inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
    )

    # Post-processing
    predicted_tokens = loader.decode_output(co_out, input_dict)
    print("The predicted token for the [MASK] is: ", predicted_tokens)


# Task-specific models like AlbertForTokenClassification are pre-trained on general datasets.
# To make them suitable for specific tasks, they need to be fine-tuned on a labeled dataset for that task.

token_classification_params = [
    pytest.param(TokenClassificationVariant.BASE_V1),
    pytest.param(TokenClassificationVariant.LARGE_V1),
    pytest.param(TokenClassificationVariant.XLARGE_V1),
    pytest.param(TokenClassificationVariant.XXLARGE_V1),
    pytest.param(TokenClassificationVariant.BASE_V2),
    pytest.param(TokenClassificationVariant.LARGE_V2),
    pytest.param(TokenClassificationVariant.XLARGE_V2),
    pytest.param(TokenClassificationVariant.XXLARGE_V2),
]


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
    input_dict = loader.load_inputs()
    inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    if variant == TokenClassificationVariant.XXLARGE_V2:
        pcc = 0.87
    else:
        pcc = 0.95

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # Post-processing
    predicted_tokens_classes = loader.decode_output(co_out, input_dict)

    print(f"Context: {loader.sample_text}")
    print(f"Answer: {predicted_tokens_classes}")


@pytest.mark.nightly
@pytest.mark.parametrize("variant", [QuestionAnsweringVariant.SQUAD2])
def test_albert_question_answering_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.ALBERT,
        variant=variant,
        task=Task.QA,
        source=Source.HUGGINGFACE,
    )

    # Load Model and inputs
    loader = QuestionAnsweringLoader(variant=variant)
    framework_model = loader.load_model()
    framework_model.config.return_dict = False
    input_dict = loader.load_inputs()
    inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    print("predicted answer ", loader.decode_output(co_out, input_dict))


@pytest.mark.nightly
@pytest.mark.parametrize("variant", [SequenceClassificationVariant.IMDB])
def test_albert_sequence_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.ALBERT,
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load Model and inputs
    loader = SequenceClassificationLoader(variant=variant)
    framework_model = loader.load_model()
    framework_model.config.return_dict = False
    input_dict = loader.load_inputs()
    inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    print(f"predicted category: {loader.decode_output(co_out)}")
