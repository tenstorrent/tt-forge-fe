# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import forge
import torch
import os
# import yolov5  # use this instead forge/test/tvm/cnn/pytorch/tests_C/test_yolov5.py

from ..common import benchmark_model
from forge.config import _get_global_compiler_config


@benchmark_model(configs=["s", "m"])
def yolo_v5(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str, math_fidelity: str):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_auto_transposing_placement = True

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["FORGE_RIBBON2"] = "1"

    from forge._C.backend_api import BackendDevice
    available_devices = forge.detect_available_devices()

    # Temp perf workaround for tenstorrent/bbe#2595
    os.environ["FORGE_PAD_OUTPUT_BUFFER"] = "1"

    if data_type == "Fp16_b":
        if available_devices[0] != BackendDevice.Grayskull:
            os.environ["FORGE_FORK_JOIN_BUF_QUEUES"] = "1"

    if data_type == "Bfp8_b":
        os.environ["FORGE_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"
        # Temp workaround for tenstorrent/bbe#2595, output BW is unpredictable.
        os.environ["FORGE_DISABLE_STREAM_OUTPUT"] = "1"

    if available_devices[0] == BackendDevice.Grayskull:
        compiler_cfg.enable_tm_cpu_fallback = True
        compiler_cfg.enable_tm_cpu_fallback = True
        compiler_cfg.enable_auto_fusing = False  # required to fix accuracy
        os.environ["FORGE_DECOMPOSE_SIGMOID"] = "1"

    # Set model parameters based on chosen task and model configuration
    config_name = ""
    if config == "s":
        config_name = "yolov5s"
        img_res = 320
    elif config == "m":
        config_name = "yolov5m",
        img_res = 640
    else:
        raise RuntimeError("Unknown config")

    # Configure microbatch, if none provided
    if microbatch == 0:
        microbatch = 32  # default

    # Load model
    model = torch.hub.load("ultralytics/yolov5", config_name, pretrained=True)

    # Configure model mode for training or evaluation
    if training:
        model.train()
    else:
        model.eval()

    input_shape = (microbatch, 3, img_res, img_res)
    inputs = [torch.rand(*input_shape)]
    targets = tuple()

    # The model is implemented with dynamic shapes as it supports various input sizes... Needs to be run with proper
    # input shape on CPU so that the dynamic shapes get resolved properly, before running thru forge
    model(inputs[0])

    modules = {"tt": forge.PyTorchModule(f"yolov5_{config}_{compiler_cfg.balancer_policy}", model)}

    # Add loss function, if training
    if training:
        model["cpu-loss"] = forge.PyTorchModule("l1loss", torch.nn.L1Loss())
        targets = [torch.rand(1, 100)]

    return modules, inputs, targets, {}
