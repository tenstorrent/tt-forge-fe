# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import onnx
import torch
from loguru import logger

import forge
from forge import OnnxModule
from forge.config import _get_global_compiler_config
from ...common import benchmark_model


@benchmark_model(configs=["default"])
def custom_vit_highres(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str):

    # Set up ONNX model path
    model_path = "third_party/confidential_customer_models/model_0/files/vit_b_16_w1280h800.onnx"
    if not os.path.exists(model_path):
        logger.error("Model not found! Exiting...")
        exit(1)

    # Load ONNX model
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    forge_onnx_model = OnnxModule(
        "CUSTOM_ViT_HighRes",
        onnx_model,
        model_path,
    )

    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"

    # Overrides
    os.environ["FORGE_RIBBON2"] = "1"

    models = {"tt" : forge_onnx_model}
    dimension = onnx_model.graph.input[0].type.tensor_type.shape
    input_shape = [d.dim_value for d in dimension.dim]
    inputs = [torch.rand(*input_shape)]
    targets = tuple()
    if training:
        targets = [torch.rand(1, 100)]

    return models, inputs, targets, {}
