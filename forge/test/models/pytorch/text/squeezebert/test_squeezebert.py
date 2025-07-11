# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["squeezebert/squeezebert-mnli"])
def test_squeezebert_sequence_classification_pytorch(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.SQUEEZEBERT,
        variant=variant,
        task=Task.NLP_TEXT_CLS,
        source=Source.HUGGINGFACE,
    )

    # Load Bart tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    framework_model = download_model(AutoModelForSequenceClassification.from_pretrained, variant, return_dict=False)

    # Example from multi-nli validation set
    text = """Hello, my dog is cute"""

    # Data preprocessing
    input_tokens = tokenizer.encode(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    predicted_class_id = co_out[0].argmax().item()
    predicted_category = framework_model.config.id2label[predicted_class_id]

    print(f"predicted category: {predicted_category}")
