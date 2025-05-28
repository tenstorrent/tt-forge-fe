# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import onnx
from transformers import BertModel, BertTokenizer

import forge
from forge.verify.verify import verify

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from test.utils import download_model



@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["sentence-transformers/all-MiniLM-L6-v2"])
def test_minilm_sequence_classification_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.MINILM,
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load MiniLM tokenizer and model from HuggingFace
    tokenizer = download_model(BertTokenizer.from_pretrained, variant)
    framework_model = download_model(BertModel.from_pretrained, variant, return_dict=False)

    # Load data sample
    sample_text = "This movie is great! I really enjoyed watching it."

    # Data preprocessing
    input_tokens = tokenizer(sample_text, padding=True, truncation=True, return_tensors="pt")
    inputs = [input_tokens["input_ids"]]

    # Export model to ONNX
    # TODO: Replace with pre-generated ONNX model to avoid exporting from scratch.
    onnx_path = f"{forge_tmp_path}/minilm.onnx"
    torch.onnx.export(framework_model, inputs[0], onnx_path, opset_version=17)

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
