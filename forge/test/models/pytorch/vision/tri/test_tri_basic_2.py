# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# import pybuda
from types import SimpleNamespace

import cv2
import pytest
import torch

import forge
from forge.forge_property_utils import Framework, Task
from forge.verify.verify import verify

# import sys
# sys.path.append("third_party/confidential_customer_models/internal/tri_basic_2/scripts")
# from semseg_tri import resnet34_semseg


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="dependent on CCM repo and Hang observed at post_initial_graph_pass")
@pytest.mark.nightly
def test_tri_basic_2_sematic_segmentation_pytorch(forge_property_recorder):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="tri", variant="basic_2", task=Task.SEMANTIC_SEGMENTATION
    )

    # Sample Input
    image_w = 800
    image_h = 800
    image = cv2.imread("third_party/confidential_customer_models/internal/tri_basic_2/files/samples/left.png")
    image = cv2.resize(image, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
    image_tensor = (torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0).contiguous()

    # Load the model and weights
    hparams = SimpleNamespace(num_classes=24)
    framework_model = resnet34_semseg(hparams)
    state_dict = torch.load(
        "third_party/confidential_customer_models/internal/tri_basic_2/files/weights/basic_semseg.ckpt",
        map_location="cpu",
    )
    framework_model.load_state_dict(state_dict)
    framework_model.eval()

    inputs = [image_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
