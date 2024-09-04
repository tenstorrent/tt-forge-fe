# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import forge
import torch

from ..common import benchmark_model
from forge.config import _get_global_compiler_config
from pytorchcv.model_provider import get_model as ptcv_get_model


@benchmark_model(configs=["2d", "3d"])
def openpose_osmr_body(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str, math_fidelity: str):
    # Configurations
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.enable_auto_transposing_placement = True

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["FORGE_RIBBON2"] = "1" 

    os.environ["FORGE_SUPRESS_T_FACTOR_MM"] = "13"
    os.environ["FORGE_ENABLE_HOST_INPUT_NOP_BUFFERING"] = "1"

    # Set model parameters based on chosen task and model configuration
    model_name = ""
    img_res = 224
    if config == "2d":
        model_name = "lwopenpose2d_mobilenet_cmupan_coco"
    elif config == "3d":
        model_name = "lwopenpose3d_mobilenet_cmupan_coco"
    else:
        raise RuntimeError("Unknown config")

    # Load model
    model = ptcv_get_model(model_name, pretrained=True)

    # Configure model mode for training or evaluation
    if training:
        model.train()
    else:
        model.eval()

    modules = {"tt": forge.PyTorchModule("openpose_body_" + config + "_pt", model)}

    input_shape = (microbatch, 3, img_res, img_res)
    inputs = [torch.rand(*input_shape)]
    targets = tuple()

    # Add loss function, if training
    if training:
        model["cpu-loss"] = forge.PyTorchModule("l1loss", torch.nn.L1Loss())
        targets = [torch.rand(1, 100)]

    return modules, inputs, targets, {}