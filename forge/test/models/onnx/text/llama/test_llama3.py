# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import forge
from forge.verify.verify import verify

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from test.utils import download_model
from test.models.models_utils import TextModelWrapper
import onnx


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "meta-llama/Llama-3.1-8B",
            marks=pytest.mark.skip(reason="Segmentation fault"),
        ),
        pytest.param(
            "meta-llama/Llama-3.2-1B",
            marks=pytest.mark.skip(reason="Insufficient host DRAM to run this model (requires a bit more than 31 GB"),
        ),
        pytest.param(
            "meta-llama/Llama-3.2-3B",
            marks=pytest.mark.skip(reason="Segmentation fault"),
        ),
        pytest.param(
            "meta-llama/Llama-3.1-8B-Instruct",
            marks=pytest.mark.skip(reason="Segmentation fault"),
        ),
        pytest.param(
            "meta-llama/Llama-3.2-1B-Instruct",
            marks=pytest.mark.skip(reason="Insufficient host DRAM to run this model (requires a bit more than 30 GB"),
        ),
        pytest.param(
            "meta-llama/Llama-3.2-3B-Instruct",
            marks=pytest.mark.skip(reason="Segmentation fault"),
        ),
    ],
)
def test_llama3_causal_lm_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.LLAMA3,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Load model and tokenizer
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token
    model = download_model(AutoModelForCausalLM.from_pretrained, variant, use_cache=False)
    framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
    framework_model.eval()

    # Prepare input
    input_prompt = "Hey how are you doing today?"
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True,
    )

    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]
    inputs = [input_ids, attn_mask]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/model.onnx"
    torch.onnx.export(framework_model, tuple(inputs), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)

    # passing model file instead of model proto due to size of the model(>2GB) - #https://github.com/onnx/onnx/issues/3775#issuecomment-943416925
    onnx.checker.check_model(onnx_path)
    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Compile model
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
