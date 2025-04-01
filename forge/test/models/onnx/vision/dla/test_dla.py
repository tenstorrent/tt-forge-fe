# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import onnx
import os
import requests
import pytest
import torchvision.transforms as transforms
from PIL import Image

# TODO: These are old forge, we should update them to the currently version.
# import forge
# from forge import DepricatedVerifyConfig
# from forge.verify.backend import verify_module
# from forge.verify.config import TestKind
# from forge._C.backend_api import BackendDevice


variants = [
    "dla34",
    "dla46_c",
    "dla46x_c",
    "dla60x_c",
    "dla60",
    "dla60x",
    "dla102",
    "dla102x",
    "dla102x2",
    "dla169",
]


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Requires restructuring")
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_dla_onnx(test_device, variant):
    compiler_cfg = forge.config.CompilerConfig()
    compiler_cfg.default_df_override = forge._C.Float16_b

    # Load data sample
    url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = transform(image).unsqueeze(0)

    onnx_dir_path = "dla"
    onnx_model_path = f"dla/{variant}_Opset18.onnx"
    if not os.path.exists(onnx_model_path):
        if not os.path.exists("dla"):
            os.mkdir("dla")
        url = f"https://github.com/onnx/models/raw/main/Computer_Vision/{variant}_Opset18_timm/{variant}_Opset18.onnx?download="
        response = requests.get(url, stream=True)
        with open(onnx_model_path, "wb") as f:
            f.write(response.content)

    # Load DLA model
    model_name = f"dla_{variant}_onnx"
    onnx_model = onnx.load(onnx_model_path)
    tt_model = forge.OnnxModule(model_name, onnx_model)

    pcc = 0.99
    if test_device.arch == BackendDevice.Wormhole_B0:
        if variant == "dla34":
            pcc = 0.98
        elif variant == "dla169":
            pcc = 0.96
    elif test_device.arch == BackendDevice.Grayskull:
        if variant == "dla46_c":
            pcc = 0.97
        if variant == "dla102x2":
            os.environ["FORGE_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    verify_module(
        tt_model,
        input_shapes=[img_tensor.shape],
        inputs=[(img_tensor,)],
        verify_cfg=DepricatedVerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=pcc,
        ),
    )

    # Cleanup model files
    os.remove(onnx_model_path)
    os.rmdir(onnx_dir_path)
