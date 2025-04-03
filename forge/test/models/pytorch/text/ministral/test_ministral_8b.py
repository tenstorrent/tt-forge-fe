# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import forge

from test.models.utils import Framework, Source, Task, build_module_name


# Create wrapper that returns tensor-only outputs
class TraceableModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

        # Flatten past_key_values tuple
        logits, past_key_values = outputs[0], outputs[1]
        flattened_past = []
        for layer in past_key_values:
            flattened_past.extend(layer)
        return (logits, *flattened_past)


@pytest.mark.nightly
def test_ministral(record_forge_property):
    # hf_token = ""

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="mistralai/Ministral-8B-Instruct-2410",
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("priority_1")
    forge_property_recorder.record_model_name(module_name)

    # Load model with modified configuration
    config = AutoConfig.from_pretrained("mistralai/Ministral-8B-Instruct-2410", use_auth_token=hf_token)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Ministral-8B-Instruct-2410", use_auth_token=hf_token)
    framework_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Ministral-8B-Instruct-2410", config=config, use_auth_token=hf_token
    )
    framework_model.eval()

    # Create traceable model instance
    traceable_model = TraceableModel(framework_model)

    # Generate sample inputs
    prompt = "What are the benefits of AI in healthcare?"
    input_tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = input_tokens["input_ids"]
    attention_mask = input_tokens["attention_mask"]

    # Forge compile framework model
    inputs = [input_ids, attention_mask]

    # Model Verification
    compiled_model = forge.compile(traceable_model, sample_inputs=inputs)
