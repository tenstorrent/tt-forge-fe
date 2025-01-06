# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import forge
import requests
import pytest
import torchvision.transforms as transforms
from PIL import Image
from test.models.pytorch.vision.dla.utils.dla_model import (
    dla34,
    dla46_c,
    dla46x_c,
    dla60x_c,
    dla60,
    dla60x,
    dla102,
    dla102x,
    dla102x2,
    dla169,
)
from test.models.utils import build_module_name, Framework
from forge.verify.verify import verify


variants_func = {
    "dla34": dla34,
    "dla46_c": dla46_c,
    "dla46x_c": dla46x_c,
    "dla60x_c": dla60x_c,
    "dla60": dla60,
    "dla60x": dla60x,
    "dla102": dla102,
    "dla102x": dla102x,
    "dla102x2": dla102x2,
    "dla169": dla169,
}
variants = list(variants_func.keys())


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dla_pytorch(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="dla", variant=variant)

    record_forge_property("module_name", module_name)

    func = variants_func[variant]

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

    framework_model = func(pretrained="imagenet")
    framework_model.eval()

    inputs = [img_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
