# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from PIL import Image

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name

# https://github.com/holli/yolov3_pytorch
# sys.path = list(set(sys.path + ["third_party/confidential_customer_models/model_2/pytorch/"]))

# from yolo_v3.holli_src import utils
# from yolo_v3.holli_src.yolo_layer import *
# from yolo_v3.holli_src.yolov3_tiny import *
# from yolo_v3.holli_src.yolov3 import *


def generate_model_yolotinyV3_imgcls_holli_pytorch():
    model = Yolov3Tiny(num_classes=80, use_wrong_previous_anchors=True)
    model.load_state_dict(torch.load("weights/yolov3_tiny_coco_01.h5"))
    model.eval()

    sz = 512
    imgfile = "person.jpg"
    img_org = Image.open(imgfile).convert("RGB")
    img_resized = img_org.resize((sz, sz))
    img_tensor = utils.image2torch(img_resized)

    return model, [img_tensor], {}


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_yolov3_tiny_holli_pytorch(record_forge_property):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="yolov_3",
        variant="tiny_holli_pytorch",
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCH_HUB,
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

    framework_model, inputs, _ = generate_model_yolotinyV3_imgcls_holli_pytorch()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


def generate_model_yoloV3_imgcls_holli_pytorch():
    model = Yolov3(num_classes=80)
    model.load_state_dict(
        torch.load(
            "weights/yolov3_coco_01.h5",
            map_location=torch.device("cpu"),
        )
    )
    model.eval()

    sz = 512
    imgfile = "person.jpg"
    img_org = Image.open(imgfile).convert("RGB")
    img_resized = img_org.resize((sz, sz))
    img_tensor = utils.image2torch(img_resized)

    return model, [img_tensor], {"pcc": pcc}


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_yolov3_holli_pytorch(record_forge_property):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="yolo_v3",
        variant="holli_pytorch",
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCH_HUB,
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

    framework_model, inputs, _ = generate_model_yoloV3_imgcls_holli_pytorch()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
