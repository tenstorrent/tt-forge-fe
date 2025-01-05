# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import forge
import pytest
import numpy as np
import torch
import requests
import os
from test.models.pytorch.vision.ssd300_resnet50.utils.image_utils import prepare_input
from test.models.utils import build_module_name


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_pytorch_ssd300_resnet50():
    # STEP 2 : prepare model
    model = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd", pretrained=False)
    url = "https://api.ngc.nvidia.com/v2/models/nvidia/ssd_pyt_ckpt_amp/versions/19.09.0/files/nvidia_ssdpyt_fp16_190826.pt"
    checkpoint_path = "nvidia_ssdpyt_fp16_190826.pt"

    response = requests.get(url)
    with open(checkpoint_path, "wb") as f:
        f.write(response.content)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # STEP 3 : prepare input
    img = "http://images.cocodataset.org/val2017/000000397133.jpg"
    HWC = prepare_input(img)
    CHW = np.swapaxes(np.swapaxes(HWC, 0, 2), 1, 2)
    batch = np.expand_dims(CHW, axis=0)
    input_batch = torch.from_numpy(batch).float()
    module_name = build_module_name(framework="pt", model="ssd300_resnet50")
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)
