# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import onnx
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
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

from test.utils import download_model

params = [
    "facebook/dpr-ctx_encoder-single-nq-base",
    "facebook/dpr-ctx_encoder-multiset-base",
]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", params)
def test_dpr_context_encoder_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.DPR,
        variant=variant,
        suffix="context_encoder",
        source=Source.HUGGINGFACE,
        task=Task.QA,
    )

    tokenizer = download_model(DPRContextEncoderTokenizer.from_pretrained, variant)
    framework_model = download_model(DPRContextEncoder.from_pretrained, variant, return_dict=False)

    # Load data sample
    sample_text = "Hello, is my dog cute?"

    # Data preprocessing
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"], input_tokens["token_type_ids"]]

    # Export to ONNX
    onnx_path = f"{forge_tmp_path}/dpr_context_encoder.onnx"
    torch.onnx.export(
        framework_model,
        tuple(inputs),
        onnx_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["embeddings"],
        opset_version=17,
    )

    # Load and check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Wrap ONNX model
    onnx_module = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Compile ONNX model
    compiled_model = forge.compile(onnx_module, inputs, module_name=module_name)

    # Verify compiled model
    _, co_out = verify(inputs, onnx_module, compiled_model)

    # Print embeddings
    print("Embeddings:", co_out[0])
