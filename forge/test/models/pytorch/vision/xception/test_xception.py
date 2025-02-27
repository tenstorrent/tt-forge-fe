# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import urllib

import pytest
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model


def generate_model_xception_imgcls_timm(variant):
    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(timm.create_model, variant, pretrained=True)
    framework_model.eval()

    # STEP 3: Prepare input
    config = resolve_data_config({}, model=framework_model)
    transform = create_transform(**config)
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    return framework_model, [img_tensor]


variants = ["xception", "xception41", "xception65", "xception71"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_xception_timm(record_forge_property, variant):
    if variant != "xception":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="xception",
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

    (framework_model, inputs) = generate_model_xception_imgcls_timm(variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
