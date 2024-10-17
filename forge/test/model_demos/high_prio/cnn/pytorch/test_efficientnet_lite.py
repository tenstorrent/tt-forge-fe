# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
## EfficientNet V1 demo
import pytest

# STEP 0: import Forge library
import forge
from PIL import Image
from torchvision import transforms

## https://github.com/RangiLyu/EfficientNet-Lite/
from test.model_demos.utils.cnn.pytorch.saved.efficientnet_lite import src_efficientnet_lite as efflite


#############
def get_image_tensor(wh):
    # Image processing
    tfms = transforms.Compose(
        [
            transforms.Resize(wh),
            transforms.CenterCrop(wh),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = tfms(Image.open("forge/test/model_demos/utils/cnn/pytorch/images/img.jpeg")).unsqueeze(0)
    return img_tensor


######


@pytest.mark.skip(reason="dependent on CCM repo")
def test_efficientnet_lite_0_pytorch():
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite0"
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("efficientnet_lite/weights/efficientnet_lite0.pth")
    model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = get_image_tensor(wh)
    compiled_model = forge.compile(model, sample_inputs=img_tensor)


@pytest.mark.skip(reason="dependent on CCM repo")
def test_efficientnet_lite_1_pytorch(test_device):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite1"
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("efficientnet_lite1.pth")
    model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = get_image_tensor(wh)

    compiled_model = forge.compile(model, sample_inputs=img_tensor)


@pytest.mark.skip(reason="dependent on CCM repo")
def test_efficientnet_lite_2_pytorch(test_device):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite2"
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("efficientnet_lite2.pth")
    model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = get_image_tensor(wh)
    compiled_model = forge.compile(model, sample_inputs=img_tensor)


@pytest.mark.skip(reason="dependent on CCM repo")
def test_efficientnet_lite_3_pytorch(test_device):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite3"
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("efficientnet_lite3.pth")
    model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = get_image_tensor(wh)
    compiled_model = forge.compile(model, sample_inputs=img_tensor)


@pytest.mark.skip(reason="dependent on CCM repo")
def test_efficientnet_lite_4_pytorch(test_device):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite4"
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("efficientnet_lite4.pth")
    model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = get_image_tensor(wh)
    compiled_model = forge.compile(model, sample_inputs=img_tensor)
