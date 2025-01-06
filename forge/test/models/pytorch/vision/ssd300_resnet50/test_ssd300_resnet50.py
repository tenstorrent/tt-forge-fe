# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import requests
import torch

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.ssd300_resnet50.utils.image_utils import prepare_input
from test.models.utils import Framework, build_module_name


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_pytorch_ssd300_resnet50(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="ssd300_resnet50")

    record_forge_property("module_name", module_name)

    # STEP 2 : prepare model
    framework_model = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd", pretrained=False)
    url = "https://api.ngc.nvidia.com/v2/models/nvidia/ssd_pyt_ckpt_amp/versions/19.09.0/files/nvidia_ssdpyt_fp16_190826.pt"
    checkpoint_path = "nvidia_ssdpyt_fp16_190826.pt"

    response = requests.get(url)
    with open(checkpoint_path, "wb") as f:
        f.write(response.content)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    framework_model.load_state_dict(checkpoint["model"])
    framework_model.eval()

    # STEP 3 : prepare input
    img = "http://images.cocodataset.org/val2017/000000397133.jpg"
    HWC = prepare_input(img)
    CHW = np.swapaxes(np.swapaxes(HWC, 0, 2), 1, 2)
    batch = np.expand_dims(CHW, axis=0)
    input_batch = torch.from_numpy(batch).float()
    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
