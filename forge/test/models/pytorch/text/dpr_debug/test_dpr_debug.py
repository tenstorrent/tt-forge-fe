# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import DPRReader, DPRReaderTokenizer

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.utils import download_model

variants = ["facebook/dpr-reader-single-nq-base"]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dpr_reader_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DPR,
        variant=variant,
        suffix="reader",
        source=Source.HUGGINGFACE,
        task=Task.QA,
    )

    # Load Bert tokenizer and model from HuggingFace
    # Variants: facebook/dpr-reader-single-nq-base, facebook/dpr-reader-multiset-base
    model_ckpt = variant
    tokenizer = download_model(DPRReaderTokenizer.from_pretrained, model_ckpt)
    framework_model = download_model(DPRReader.from_pretrained, model_ckpt, return_dict=False)

    # Data preprocessing
    input_tokens = tokenizer(
        questions=["What is love?"],
        titles=["Haddaway"],
        texts=["'What Is Love' is a song recorded by the artist Haddaway"],
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

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
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        answers.append(answer)

    print("Predicted Answer:", answers[0])
