# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import onnx
from test.models.pytorch.multimodal.oft.utils.oft_utils import StableDiffusionWrapper


def get_models(inputs, tmp_path, pipe):
    # Forge compile framework model
    framework_model = StableDiffusionWrapper(pipe)

    # Export model to ONNX
    onnx_path = tmp_path / "oft.onnx"
    torch.onnx.export(
        framework_model,
        inputs,
        "oft.onnx",
        input_names=["latents", "timesteps", "prompt_embeds"],
        output_names=["sample"],
        opset_version=17,
        do_constant_folding=True,
    )

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    return onnx_model
