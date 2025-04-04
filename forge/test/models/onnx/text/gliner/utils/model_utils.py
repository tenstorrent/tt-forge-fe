# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


def export_onnx(inputs, framework_model, onnx_path):
    input_names = ["input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "words_mask": {0: "batch_size", 1: "sequence_length"},
        "text_lengths": {0: "batch_size", 1: "value"},
        "span_idx": {0: "batch_size", 1: "num_spans", 2: "idx"},
        "span_mask": {0: "batch_size", 1: "num_spans"},
        "logits": {0: "batch_size", 1: "sequence_length", 2: "num_spans", 3: "num_classes"},
    }
    torch.onnx.export(
        framework_model.model,
        inputs,
        f=onnx_path,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=12,
    )
