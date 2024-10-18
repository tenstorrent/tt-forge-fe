# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import forge
import os
import torch
import pytest
import torch.nn.functional as F
import numpy as np
from PIL import Image
import sys

# sys.path.append("forge/test/model_demos/models")
# from fchardnet import get_model, fuse_bn_recursively


@pytest.mark.skip(reason="dependent on CCM repo")
def test_fchardnet(test_device):
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    # Load and pre-process image
    image_path = "tt-forge-fe/forge/test/model_demos/high_prio/cnn/pytorch/model2/pytorch/pidnet/image/road_scenes.png"
    img = Image.open(image_path)
    img = np.array(img.resize((320, 320)), dtype=np.uint8)
    img = img[:, :, ::-1]
    mean = np.array([0.406, 0.456, 0.485]) * 255
    std = np.array([0.225, 0.224, 0.229]) * 255
    img = (img.astype(np.float64) - mean) / std
    img = torch.tensor(img).float().permute(2, 0, 1)
    input_image = img.unsqueeze(0)

    # Load model
    device = torch.device("cpu")
    arch = {"arch": "hardnet"}
    model = get_model(arch, 19).to(device)
    model = fuse_bn_recursively(model)
    model.eval()

    compiled_model = forge.compile(model, sample_inputs=[input_image])
