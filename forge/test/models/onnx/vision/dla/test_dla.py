# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import onnx
import os
import requests
import pytest
import torchvision.transforms as transforms
from PIL import Image

from test.models.pytorch.vision.dla.model_utils.utils import post_processing
import forge
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties


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


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_dla_onnx(variant, tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.DLA,
        variant=variant,
        task=Task.VISUAL_BACKBONE,
        source=Source.TORCHVISION,
    )

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
    inputs = [img_tensor]

    onnx_path = f"{tmp_path}/dla_{variant}_Opset18.onnx"
    if not os.path.exists(onnx_path):
        if not os.path.exists("dla"):
            os.mkdir("dla")
        url = f"https://github.com/onnx/models/raw/main/Computer_Vision/{variant}_Opset18_timm/{variant}_Opset18.onnx?download="
        response = requests.get(url, stream=True)
        with open(onnx_path, "wb") as f:
            f.write(response.content)

    # Load DLA model
    model_name = f"dla_{variant}_onnx"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(model_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    post_processing(co_out)
