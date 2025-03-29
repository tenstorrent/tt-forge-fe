# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import onnx
from transformers import (
    BertForMaskedLM,
    BertForQuestionAnswering,
    BertTokenizer,
)

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model


# Opset 9 is the minimum version to support BERT in Torch.
# Opset 17 is the maximum version in Torchscript.
opset_versions = [9, 17]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["bert-base-uncased"])
@pytest.mark.parametrize("opset_version", opset_versions, ids=opset_versions)
def test_bert_masked_lm_onnx(forge_property_recorder, variant, tmp_path, opset_version):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.ONNX,
        model="bert",
        variant=variant,
        task=Task.MASKED_LM,
        source=Source.HUGGINGFACE,
    )
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load Bert tokenizer and model from HuggingFace
    tokenizer = download_model(BertTokenizer.from_pretrained, variant)
    framework_model = download_model(BertForMaskedLM.from_pretrained, variant, return_dict=False)

    # Load data sample
    sample_text = "The capital of France is [MASK]."

    # Data preprocessing
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"]]

    # Export model to ONNX
    # TODO: Replace with pre-generated ONNX model to avoid exporting from scratch.
    onnx_path = f"{tmp_path}/bert_masked_lm.onnx"
    torch.onnx.export(framework_model, inputs[0], onnx_path, opset_version=opset_version)

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("opset_version", opset_versions, ids=opset_versions)
def test_bert_question_answering_onnx(forge_property_recorder, variant, tmp_path, opset_version):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.ONNX,
        model="bert",
        variant=variant,
        task=Task.QA,
        source=Source.HUGGINGFACE,
    )
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load Bert tokenizer and model from HuggingFace
    tokenizer = download_model(BertTokenizer.from_pretrained, variant)
    framework_model = download_model(BertForQuestionAnswering.from_pretrained, variant, return_dict=False)

    # Load data sample from SQuADv1.1
    context = """Super Bowl 50 was an American football game to determine the champion of the National Football League
    (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the
    National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title.
    The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
    As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed
    initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals
    (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently
    feature the Arabic numerals 50."""

    question = "Which NFL team represented the AFC at Super Bowl 50?"

    # Data preprocessing
    input_tokens = tokenizer(
        question,
        context,
        max_length=384,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"]]

    # Export model to ONNX
    # TODO: Replace with pre-generated ONNX model to avoid exporting from scratch.
    onnx_path = f"{tmp_path}/bert_qa.onnx"
    torch.onnx.export(framework_model, inputs[0], onnx_path, opset_version=opset_version)

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
