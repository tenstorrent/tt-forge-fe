# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


def export_onnx(inputs, framework_model, onnx_path):
    input_names = ["input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask"]
    torch.onnx.export(
        framework_model.model,
        inputs,
        f=onnx_path,
        input_names=input_names,
        output_names=["logits"],
        opset_version=17,
    )


def prepare_inputs(framework_model, text, labels):
    inputs, _ = framework_model.prepare_model_inputs(text, labels)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    words_mask = inputs["words_mask"]
    text_lengths = inputs["text_lengths"]
    span_idx = inputs["span_idx"]
    span_mask = inputs["span_mask"]
    inputs = [input_ids, attention_mask, words_mask, text_lengths, span_idx, span_mask]

    return inputs
