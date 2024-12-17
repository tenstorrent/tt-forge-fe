# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import forge
from forge.verify.verify import verify

from third_party.TT_AI_Example.sfa.models.model_utils import create_model
from third_party.TT_AI_Example.sfa.test_cam import parse_test_configs


def test_cam_inference():
    # Compiler configurations
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False

    # Load CAM model
    configs = parse_test_configs()

    model = create_model(configs)
    model.load_state_dict(torch.load(configs.pretrained_path))

    input_image = torch.rand(configs.input_size)
    inputs = [input_image]

    # Compile the model
    compiled_model = forge.compile(framework_model, input_image)


if __name__ == "__main__":
    test_cam_inference()
