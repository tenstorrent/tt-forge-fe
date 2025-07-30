# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

# Import the model loaders and variants from the new location
from third_party.tt_forge_models.distilbert.masked_lm.pytorch.loader import (
    ModelLoader as MaskedLMLoader,
)
from third_party.tt_forge_models.distilbert.masked_lm.pytorch.loader import (
    ModelVariant as MaskedLMVariant,
)
from third_party.tt_forge_models.distilbert.question_answering.pytorch.loader import (
    ModelLoader as QuestionAnsweringLoader,
)
from third_party.tt_forge_models.distilbert.question_answering.pytorch.loader import (
    ModelVariant as QuestionAnsweringVariant,
)
from third_party.tt_forge_models.distilbert.sequence_classification.pytorch.loader import (
    ModelLoader as SequenceClassificationLoader,
)
from third_party.tt_forge_models.distilbert.sequence_classification.pytorch.loader import (
    ModelVariant as SequenceClassificationVariant,
)
from third_party.tt_forge_models.distilbert.token_classification.pytorch.loader import (
    ModelLoader as TokenClassificationLoader,
)
from third_party.tt_forge_models.distilbert.token_classification.pytorch.loader import (
    ModelVariant as TokenClassificationVariant,
)
from transformers.modeling_outputs import (
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify


# Wrapper to return tensor outputs
class DistilBertWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Case 1: Question Answering
        if isinstance(output, QuestionAnsweringModelOutput):
            return output.start_logits, output.end_logits

        # Case 2: Token Classification, MaskedLM, Sequence Classification
        if isinstance(output, (TokenClassifierOutput, MaskedLMOutput, SequenceClassifierOutput)):
            return output.logits


variants = [
    pytest.param(MaskedLMVariant.DISTILBERT_BASE_CASED, marks=[pytest.mark.push]),
    MaskedLMVariant.DISTILBERT_BASE_UNCASED,
    MaskedLMVariant.DISTILBERT_BASE_MULTILINGUAL_CASED,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_distilbert_masked_lm_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DISTILBERT,
        variant=variant.value,
        task=Task.MASKED_LM,
        source=Source.HUGGINGFACE,
    )

    # Load model using the new loader
    loader = MaskedLMLoader(variant=variant)
    framework_model = loader.load_model()
    framework_model = DistilBertWrapper(framework_model)

    # Get sample inputs from the loader
    input_tokens = loader.load_inputs()
    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post-processing
    loader.decode_output(co_out)


variants = [QuestionAnsweringVariant.DISTILBERT_BASE_CASED_DISTILLED_SQUAD]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_distilbert_question_answering_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DISTILBERT,
        variant=variant.value,
        task=Task.QA,
        source=Source.HUGGINGFACE,
    )

    # Load model using the new loader
    loader = QuestionAnsweringLoader(variant=variant)
    framework_model = loader.load_model()
    framework_model = DistilBertWrapper(framework_model)

    # Get sample inputs from the loader
    input_tokens = loader.load_inputs()
    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    loader.decode_output(co_out)


variants = [SequenceClassificationVariant.DISTILBERT_BASE_UNCASED_FINETUNED_SST_2_ENGLISH]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_distilbert_sequence_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DISTILBERT,
        variant=variant.value,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load model using the new loader
    loader = SequenceClassificationLoader(variant=variant)
    framework_model = loader.load_model()
    framework_model = DistilBertWrapper(framework_model)

    # Get sample inputs from the loader
    input_tokens = loader.load_inputs()
    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    loader.decode_output(co_out, framework_model=framework_model)


variants = [TokenClassificationVariant.DAVLAN_DISTILBERT_BASE_MULTILINGUAL_CASED_NER_HRL]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_distilbert_token_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DISTILBERT,
        variant=variant.value,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load model using the new loader
    loader = TokenClassificationLoader(variant=variant)
    framework_model = loader.load_model()
    framework_model = DistilBertWrapper(framework_model)

    # Get sample inputs from the loader
    input_tokens = loader.load_inputs()
    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    loader.decode_output(co_out, framework_model=framework_model)
