# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
## https://github.com/RangiLyu/EfficientNet-Lite/
import pytest

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.efficientnet.utils import (
    src_efficientnet_lite as efflite,
)
from test.models.pytorch.vision.utils.utils import load_timm_model_and_input


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_0_pytorch(forge_property_recorder):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="efficientnet",
        variant="lite_0",
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

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


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_1_pytorch(forge_property_recorder):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="efficientnet",
        variant="lite_1",
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

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


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_2_pytorch(forge_property_recorder):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="efficientnet",
        variant="lite_2",
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

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


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_3_pytorch(forge_property_recorder):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="efficientnet",
        variant="lite_3",
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

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


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_4_pytorch(forge_property_recorder):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="efficientnet",
        variant="lite_4",
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

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


variants = [
    "tf_efficientnet_lite0.in1k",
    "tf_efficientnet_lite1.in1k",
    "tf_efficientnet_lite2.in1k",
    "tf_efficientnet_lite3.in1k",
    "tf_efficientnet_lite4.in1k",
]


@pytest.mark.parametrize("variant", variants)
def test_efficientnet_lite_timm(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="efficientnet_lite",
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load the model and inputs
    framework_model, inputs = load_timm_model_and_input(variant)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
