# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import cv2
import numpy as np
import torch
from test.models.utils import build_module_name, Framework


import forge

# sys.path.append("tt-forge-fe/forge/test/model_demos/high_prio/cnn/pytorch/model2/pytorch/pidnet/model")
# from model_pidnet import update_model_config, get_seg_model

variants = ["pidnet_s", "pidnet_m", "pidnet_l"]


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_pidnet_pytorch(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="pidnet", variant=variant)

    record_forge_property("module_name", module_name)

    # Load and pre-process image
    image_path = "tt-forge-fe/forge/test/model_demos/high_prio/cnn/pytorch/model2/pytorch/pidnet/image/road_scenes.png"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image -= mean
    image /= std
    image = image.transpose((2, 0, 1))
    input_image = torch.unsqueeze(torch.tensor(image), 0)

    # Load model
    cfg_model_pretrained, cfg_model_state_file = update_model_config(variant)
    model = get_seg_model(variant, cfg_model_pretrained, imgnet_pretrained=True)
    pretrained_dict = torch.load(cfg_model_state_file, map_location=torch.device("cpu"))

    if "state_dict" in pretrained_dict:
        pretrained_dict = pretrained_dict["state_dict"]
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()

    compiled_model = forge.compile(model, sample_inputs=[input_image], module_name=module_name)
