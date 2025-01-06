# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
## https://github.com/RangiLyu/EfficientNet-Lite/
import pytest

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.efficientnet.utils import (
    src_efficientnet_lite as efflite,
)
from test.models.utils import Framework, build_module_name


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_0_pytorch(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="efficientnet", variant="lite_0")

    record_forge_property("module_name", module_name)

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite0"
    framework_model = efflite.build_efficientnet_lite(model_name, 1000)
    framework_model.load_pretrain("efficientnet_lite/weights/efficientnet_lite0.pth")
    framework_model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = efflite.get_image_tensor(wh)
    inputs = [img_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_1_pytorch(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="efficientnet", variant="lite_1")

    record_forge_property("module_name", module_name)

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite1"
    framework_model = efflite.build_efficientnet_lite(model_name, 1000)
    framework_model.load_pretrain("efficientnet_lite1.pth")
    framework_model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = efflite.get_image_tensor(wh)
    inputs = [img_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_2_pytorch(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="efficientnet", variant="lite_2")

    record_forge_property("module_name", module_name)

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite2"
    framework_model = efflite.build_efficientnet_lite(model_name, 1000)
    framework_model.load_pretrain("efficientnet_lite2.pth")
    framework_model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = efflite.get_image_tensor(wh)
    inputs = [img_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_3_pytorch(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="efficientnet", variant="lite_3")

    record_forge_property("module_name", module_name)

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite3"
    framework_model = efflite.build_efficientnet_lite(model_name, 1000)
    framework_model.load_pretrain("efficientnet_lite3.pth")
    framework_model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = efflite.get_image_tensor(wh)
    inputs = [img_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_4_pytorch(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="efficientnet", variant="lite_4")

    record_forge_property("module_name", module_name)

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite4"
    framework_model = efflite.build_efficientnet_lite(model_name, 1000)
    framework_model.load_pretrain("efficientnet_lite4.pth")
    framework_model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = efflite.get_image_tensor(wh)
    inputs = [img_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
