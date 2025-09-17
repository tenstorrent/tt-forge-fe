# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import onnx
from transformers import (
    DistilBertForTokenClassification,
    DistilBertTokenizer,
)

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
from test.models.onnx.text.distilbert.model_utils.model_utils import DistilBertWrapper


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", ["Davlan/distilbert-base-multilingual-cased-ner-hrl"])
def test_distilbert_token_classification_pytorch(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.DISTILBERT,
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load DistilBERT tokenizer and model from HuggingFace
    tokenizer = download_model(DistilBertTokenizer.from_pretrained, variant)
    framework_model = download_model(DistilBertForTokenClassification.from_pretrained, variant)
    framework_model = DistilBertWrapper(framework_model)

    # Load data sample
    sample_text = "HuggingFace is a company based in Paris and New York"

    # Data preprocessing
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Export to ONNX
    variant_name = variant.replace("/", "_")
    onnx_path = f"{forge_tmp_path}/{variant_name}.onnx"
    torch.onnx.export(
        framework_model,
        tuple(inputs),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=17,
    )

    # Load ONNX model
    onnx.checker.check_model(onnx_path)
    onnx_model = onnx.load(onnx_path)
    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Compile with Forge
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Run verification
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # post processing
    predicted_token_class_ids = co_out[0].argmax(-1)
    predicted_token_class_ids = torch.masked_select(predicted_token_class_ids, input_tokens["attention_mask"][0] == 1)

    # Decode into readable text
    predicted_text = tokenizer.decode(predicted_token_class_ids.tolist(), skip_special_tokens=True)

    print(f"Context: {sample_text}")
    print(f"Answer: {predicted_text}")
