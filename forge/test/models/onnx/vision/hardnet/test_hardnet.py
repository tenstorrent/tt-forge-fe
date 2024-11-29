# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import forge, os
import onnx
from PIL import Image
from torchvision import transforms
import urllib

# from forge.verify.backend import verify_module
import pytest

# from forge import VerifyConfig
# from forge.verify.config import TestKind
# from forge._C.backend_api import BackendDevice

variants = ["hardnet68", "hardnet85", "hardnet68ds", "hardnet39ds"]


@pytest.mark.skip(reason="Not supported")
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_hardnet_onnx(variant, test_device):

    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = forge.DataFormat.Float16_b
    os.environ["FORGE_RIBBON2"] = "1"

    if variant == "hardnet68ds":
        os.environ["FORGE_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    if variant == "hardnet85" and test_device.arch == BackendDevice.Grayskull:
        os.environ["FORGE_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    # Download an example image
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    # Preprocessing
    input_image = Image.open(filename)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    img_tensor = input_tensor.unsqueeze(0)

    load_path = f"third_party/confidential_customer_models/generated/files/{variant}.onnx"
    model_name = f"{variant}_onnx"

    # Create Forge module from onnx weights
    model = onnx.load(load_path)
    tt_model = forge.OnnxModule(model_name, model, load_path)

    # Run inference on Tenstorrent device
    verify_module(
        tt_model,
        input_shapes=([img_tensor.shape]),
        inputs=([img_tensor]),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.98,
        ),
    )
