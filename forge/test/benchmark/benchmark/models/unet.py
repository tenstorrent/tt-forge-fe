# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import forge
import torch

from ..common import benchmark_model
from forge.config import _get_global_compiler_config


@benchmark_model(configs=["256"])
def unet(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str, math_fidelity: str):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.enable_auto_transposing_placement = True

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["FORGE_RIBBON2"] = "1"

    # Manually enable amp light for Ribbon
    if compiler_cfg.balancer_policy == "Ribbon":
        compiler_cfg.enable_amp_light()

    os.environ["FORGE_ENABLE_HOST_INPUT_NOP_BUFFERING"] = "1"
    os.environ["FORGE_ALLOW_MULTICOLUMN_SPARSE_MATMUL"] = "1"
    os.environ["FORGE_SUPRESS_T_FACTOR_MM"] = "60"

    # These are about to be enabled by default.
    #
    os.environ["FORGE_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"

    # Set model parameters based on chosen task and model configuration
    if config == "256":
        model = torch.hub.load(
            "mateuszforge/brain-segmentation-pytorch",
            "unet",
            in_channels=3,
            out_channels=1,
            init_features=32,
            pretrained=True,
        )
        img_res = 256
    else:
        raise RuntimeError("Unknown config")

    # Configure model mode for training or evaluation
    if training:
        model.train()
    else:
        model.eval()

    modules = {"tt": forge.PyTorchModule(f"th_unet_{config}_{compiler_cfg.balancer_policy}", model)}

    input_shape = (microbatch, 3, img_res, img_res)
    inputs = [torch.rand(*input_shape)]
    targets = tuple()

    # Add loss function, if training
    if training:
        model["cpu-loss"] = forge.PyTorchModule("l1loss", torch.nn.L1Loss())
        targets = [torch.rand(1, 100)]

    return modules, inputs, targets, {}
