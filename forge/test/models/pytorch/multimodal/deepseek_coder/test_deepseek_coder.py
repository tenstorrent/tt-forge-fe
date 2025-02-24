# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import forge
from forge.verify.verify import verify

from test.models.pytorch.multimodal.deepseek_coder.utils.load_model import (
    DeepSeekWrapper,
    download_model_and_tokenizer,
    generate_no_cache,
)
from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.parametrize("variant", ["deepseek-coder-1.3b-instruct"])
def test_deepseek_inference_no_cache_cpu(variant):
    model_name = f"deepseek-ai/{variant}"
    model, tokenizer, inputs = download_model_and_tokenizer(model_name)

    framework_model = DeepSeekWrapper(model)

    generated_text = generate(max_new_tokens=200, model=framework_model, inputs=inputs, tokenizer=tokenizer)
    print(generated_text)


@pytest.mark.parametrize("variant", ["deepseek-coder-1.3b-instruct"])
def test_deepseek_inference_no_cache(record_forge_property, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="deepseek", variant=variant, task=Task.QA, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # Load Model and Tokenizer
    model_name = f"deepseek-ai/{variant}"
    model, tokenizer, inputs = download_model_and_tokenizer(model_name)
    framework_model = DeepSeekWrapper(model)
    batch_size, seq_len = inputs.shape
    max_new_tokens = 200
    max_seq_len = seq_len + max_new_tokens
    padded_inputs = torch.randint(low=0, high=1024, size=(1, max_seq_len), dtype=torch.int64)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=[padded_inputs], module_name=module_name)
    padded_inputs[:, :seq_len] = inputs

    # Model Verification
    verify([padded_inputs], framework_model, compiled_model)

    generated_text = generate_no_cache(max_new_tokens=200, model=compiled_model, inputs=inputs, tokenizer=tokenizer)
    print(generated_text)
