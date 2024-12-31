# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

from PIL import Image
import os
import torch
import forge
from forge.test.models.utils import build_module_name

# https://github.com/holli/yolov3_pytorch
# sys.path = list(set(sys.path + ["third_party/confidential_customer_models/model_2/pytorch/"]))

# from yolo_v3.holli_src import utils
# from yolo_v3.holli_src.yolo_layer import *
# from yolo_v3.holli_src.yolov3_tiny import *
# from yolo_v3.holli_src.yolov3 import *


def generate_model_yolotinyV3_imgcls_holli_pytorch(test_device, variant):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    model = Yolov3Tiny(num_classes=80, use_wrong_previous_anchors=True)
    model.load_state_dict(torch.load("weights/yolov3_tiny_coco_01.h5"))
    model.eval()

    sz = 512
    imgfile = "person.jpg"
    img_org = Image.open(imgfile).convert("RGB")
    img_resized = img_org.resize((sz, sz))
    img_tensor = utils.image2torch(img_resized)

    return model, [img_tensor], {}


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_yolov3_tiny_holli_pytorch(test_device):
    model, inputs, _ = generate_model_yolotinyV3_imgcls_holli_pytorch(
        test_device,
        None,
    )
    module_name = build_module_name(framework="pt", model="yolov_3", variant="tiny_holli_pytorch")
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name=module_name)


def generate_model_yoloV3_imgcls_holli_pytorch(test_device, variant):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH
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


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_yolov3_holli_pytorch(test_device):
    model, inputs, other = generate_model_yoloV3_imgcls_holli_pytorch(
        test_device,
        None,
    )

    module_name = build_module_name(framework="pt", model="yolo_v3", variant="holli_pytorch")
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name=module_name)
