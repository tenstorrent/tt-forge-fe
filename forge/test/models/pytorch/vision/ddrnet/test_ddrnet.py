# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import requests
import torch
from PIL import Image
from torchvision import transforms

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Task, build_module_name

# sys.path.append("third_party/confidential_customer_models/generated/scripts/")
# from model_ddrnet import DualResNet_23, DualResNet_39, BasicBlock

# sys.path.append("third_party/confidential_customer_models/cv_demos/ddrnet/semantic_segmentation/model")
# from semseg import DualResNet, BasicBlock_seg

variants = ["ddrnet23s", "ddrnet23", "ddrnet39"]


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_ddrnet_pytorch(record_forge_property, variant):
    if variant != "ddrnet23s":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="ddrnet", variant=variant)

    # Record Forge Property
    record_forge_property("module_name", module_name)

    # STEP 2: Create Forge module from PyTorch model
    if variant == "ddrnet23s":

        framework_model = DualResNet_23(block=BasicBlock, layers=[2, 2, 2, 2], planes=32, last_planes=1024)

    elif variant == "ddrnet23":

        framework_model = DualResNet_23(block=BasicBlock, layers=[2, 2, 2, 2], planes=64, last_planes=2048)

    elif variant == "ddrnet39":

        framework_model = DualResNet_39(block=BasicBlock, layers=[3, 4, 6, 3], planes=64, last_planes=2048)

    state_dict_path = f"third_party/confidential_customer_models/generated/files/{variant}.pth"

    state_dict = torch.load(state_dict_path, map_location=torch.device("cpu"))

    framework_model.load_state_dict(state_dict, strict=False)
    framework_model.eval()

    # STEP 3: Prepare input
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    input_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants = ["ddrnet23s_cityscapes", "ddrnet23_cityscapes"]


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_ddrnet_semantic_segmentation_pytorch(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="ddrnet", variant=variant, task=Task.SEMANTIC_SEGMENTATION
    )

    # Record Forge Property
    record_forge_property("module_name", module_name)

    # prepare model
    if variant == "ddrnet23s_cityscapes":
        framework_model = DualResNet(
            BasicBlock_seg,
            [2, 2, 2, 2],
            num_classes=19,
            planes=32,
            spp_planes=128,
            head_planes=64,
            augment=True,
        )

    elif variant == "ddrnet23_cityscapes":
        framework_model = DualResNet(
            BasicBlock_seg,
            [2, 2, 2, 2],
            num_classes=19,
            planes=64,
            spp_planes=128,
            head_planes=128,
            augment=True,
        )

    state_dict_path = (
        f"third_party/confidential_customer_models/cv_demos/ddrnet/semantic_segmentation/weights/{variant}.pth"
    )
    state_dict = torch.load(state_dict_path, map_location=torch.device("cpu"))
    framework_model.load_state_dict(state_dict, strict=False)
    framework_model.eval()

    # prepare input
    image_path = "third_party/confidential_customer_models/cv_demos/ddrnet/semantic_segmentation/image/road_scenes.png"
    input_image = Image.open(image_path)
    input_tensor = transforms.ToTensor()(input_image)
    input_batch = input_tensor.unsqueeze(0)

    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
