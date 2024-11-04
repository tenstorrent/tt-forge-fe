# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# import pybuda
import torch
import forge
from types import SimpleNamespace
import pytest
import cv2
import os

# import sys
# sys.path.append("third_party/confidential_customer_models/internal/tri_basic_2/scripts")
# from semseg_tri import resnet34_semseg


@pytest.mark.skip(reason="dependent on CCM repo and Hang observed at post_initial_graph_pass")
def test_tri_basic_2_sematic_segmentation_pytorch(test_device):

    # Set PyBuda configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.GENERATE_INITIAL_GRAPH

    # Sample Input
    image_w = 800
    image_h = 800
    image = cv2.imread("third_party/confidential_customer_models/internal/tri_basic_2/files/samples/left.png")
    image = cv2.resize(image, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
    image_tensor = (torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0).contiguous()

    # Load the model and weights
    hparams = SimpleNamespace(num_classes=24)
    model = resnet34_semseg(hparams)
    state_dict = torch.load(
        "third_party/confidential_customer_models/internal/tri_basic_2/files/weights/basic_semseg.ckpt",
        map_location="cpu",
    )
    model.load_state_dict(state_dict)
    model.eval()

    print("type(image_tensor)", type(image_tensor))
    inputs = image_tensor
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_tri_basic_2_semseg")
