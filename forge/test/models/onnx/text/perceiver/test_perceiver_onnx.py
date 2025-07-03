# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/deepmind/language-perceiver

import pytest
from transformers import PerceiverForMaskedLM, PerceiverTokenizer

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
import torch
import onnx


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["deepmind/language-perceiver"])
def test_perceiverio_masked_lm_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.PERCEIVERIO,
        variant=variant,
        task=Task.MASKED_LM,
        source=Source.HUGGINGFACE,
    )

    # Load model and tokenizer
    tokenizer = download_model(PerceiverTokenizer.from_pretrained, variant)
    torch_model = download_model(PerceiverForMaskedLM.from_pretrained, variant, return_dict=False)
    torch_model.eval()

    # Prepare input
    text = "This is an incomplete sentence where some words are missing."
    encoding = tokenizer(text, padding="max_length", return_tensors="pt")
    encoding.input_ids[0, 52:61] = tokenizer.mask_token_id
    inputs = [encoding.input_ids, encoding.attention_mask]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/" + str(variant).split("/")[-1].replace("-", "_") + ".onnx"
    torch.onnx.export(torch_model, (inputs[0], inputs[1]), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # post processing
    logits = co_out[0]
    masked_tokens_predictions = logits[0, 51:61].argmax(dim=-1)
    print(f"The predicted token for the [MASK] is: {tokenizer.decode(masked_tokens_predictions)}")
