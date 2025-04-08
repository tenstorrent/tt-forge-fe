# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import onnx
from transformers import BertModel, BertTokenizer

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model


# Opset 9 is the minimum version to support BERT-like models in Torch.
# Opset 17 is the maximum version in Torchscript.
opset_versions = [9, 17]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["sentence-transformers/all-MiniLM-L6-v2"])
@pytest.mark.parametrize("opset_version", opset_versions, ids=opset_versions)
def test_minilm_sequence_classification_onnx(forge_property_recorder, variant, tmp_path, opset_version):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.ONNX,
        model="minilm",
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

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
    onnx_path = f"{tmp_path}/minilm.onnx"
    torch.onnx.export(framework_model, inputs[0], onnx_path, opset_version=opset_version)

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
