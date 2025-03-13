# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name

# sys.path = list(set(sys.path + ["third_party/confidential_customer_models/model_2/pytorch/"]))
# from mobilenetv1_ssd.vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_mobilenet_v1_ssd_pytorch_1x1(forge_property_recorder):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="mobilenet",
        variant="ssd",
        source=Source.TORCHVISION,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load PASCAL VOC dataset class labels
    label_path = "mobilenetv1_ssd/models/voc-model-labels.txt"
    class_names = [name.strip() for name in open(label_path).readlines()]
    number_of_classes = len(class_names)

    # STEP 2: Create Forge module from PyTorch model
    model_path = "mobilenetv1_ssd/models/mobilenet-v1-ssd-mp-0_675.pth"
    framework_model = create_mobilenetv1_ssd(number_of_classes)
    framework_model.load(model_path)
    framework_model.eval()

    input_shape = (1, 3, 300, 300)
    inputs = [torch.rand(input_shape)]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
