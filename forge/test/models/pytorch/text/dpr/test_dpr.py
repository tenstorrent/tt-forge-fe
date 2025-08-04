# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from third_party.tt_forge_models.dpr.context_encoder.pytorch.loader import (
    ModelLoader as ContextEncoderLoader,
)
from third_party.tt_forge_models.dpr.context_encoder.pytorch.loader import (
    ModelVariant as ContextEncoderVariant,
)
from third_party.tt_forge_models.dpr.question_encoder.pytorch.loader import (
    ModelLoader as QuestionEncoderLoader,
)
from third_party.tt_forge_models.dpr.question_encoder.pytorch.loader import (
    ModelVariant as QuestionEncoderVariant,
)
from third_party.tt_forge_models.dpr.reader.pytorch.loader import (
    ModelLoader as ReaderLoader,
)
from third_party.tt_forge_models.dpr.reader.pytorch.loader import (
    ModelVariant as ReaderVariant,
)

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.config import VerifyConfig
from forge.verify.verify import verify

params = [
    ContextEncoderVariant.DPR_SINGLE_NQ_BASE,
    ContextEncoderVariant.DPR_MULTISET_BASE,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", params)
def test_dpr_context_encoder_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DPR,
        variant=variant.value,
        suffix="context_encoder",
        source=Source.HUGGINGFACE,
        task=Task.QA,
    )

    # Load model using the new loader
    loader = ContextEncoderLoader(variant=variant)
    framework_model = loader.load_model()

    # Get sample inputs from the loader
    input_tokens = loader.load_inputs()
    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"], input_tokens["token_type_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # Results
    print("embeddings", co_out)


params = [
    QuestionEncoderVariant.DPR_SINGLE_NQ_BASE,
    QuestionEncoderVariant.DPR_MULTISET_BASE,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", params)
def test_dpr_question_encoder_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DPR,
        variant=variant.value,
        suffix="question_encoder",
        source=Source.HUGGINGFACE,
        task=Task.QA,
    )

    # Load model using the new loader
    loader = QuestionEncoderLoader(variant=variant)
    framework_model = loader.load_model()

    # Get sample inputs from the loader
    input_tokens = loader.load_inputs()
    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"], input_tokens["token_type_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    verify_values = True

    if variant == QuestionEncoderVariant.DPR_MULTISET_BASE:
        verify_values = False

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(verify_values=verify_values),
    )

    # Results
    print("embeddings", co_out)


variants = [ReaderVariant.DPR_SINGLE_NQ_BASE, ReaderVariant.DPR_MULTISET_BASE]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_dpr_reader_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DPR,
        variant=variant.value,
        suffix="reader",
        source=Source.HUGGINGFACE,
        task=Task.QA,
    )

    # Load model using the new loader
    loader = ReaderLoader(variant=variant)
    framework_model = loader.load_model()

    # Get sample inputs from the loader
    input_tokens = loader.load_inputs()
    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # Post processing
    start_logits = co_out[0]
    end_logits = co_out[1]
    relevance_logits = co_out[2]

    start_indices = torch.argmax(start_logits, dim=1)
    end_indices = torch.argmax(end_logits, dim=1)

    answers = []
    for i, input_id in enumerate(inputs[0]):
        start_idx = start_indices[i].item()
        end_idx = end_indices[i].item()
        answer_tokens = input_id[start_idx : end_idx + 1]
        answer = loader.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        answers.append(answer)

    print("Predicted Answer:", answers[0])
