# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import onnx
from loguru import logger
from PIL import Image
from pytorchcv.model_provider import get_model as ptcv_get_model
from third_party.tt_forge_models.tools.utils import get_file
from torchvision import transforms

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import VerifyConfig, verify

from test.models.models_utils import print_cls_results
from test.utils import download_model

variants = [
    pytest.param("vgg11", marks=pytest.mark.pr_models_regression),
    pytest.param("vgg13"),
    pytest.param("vgg16"),
    pytest.param("vgg19"),
    pytest.param("bn_vgg19", marks=pytest.mark.pr_models_regression),
    pytest.param("bn_vgg19b"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_vgg_osmr_pytorch(variant, forge_tmp_path):

    pcc = 0.99
    if variant == "vgg19":
        pcc = 0.98
    elif variant == "bn_vgg19b":
        pcc = 0.95

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.VGG,
        variant=variant,
        source=Source.OSMR,
        task=Task.OBJECT_DETECTION,
    )

    framework_model = download_model(ptcv_get_model, variant, pretrained=True)
    framework_model.eval()

    # Image preprocessing
    try:
        file_path = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
        input_image = Image.open(file_path).convert("RGB")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        input_batch = torch.rand(1, 3, 224, 224)

    inputs = [input_batch]

    # Export to ONNX
    onnx_path = f"{forge_tmp_path}/{variant}.onnx"
    torch.onnx.export(
        framework_model,
        input_batch,
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
    )

    # Load and verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx_module = forge.OnnxModule(module_name, onnx_model)

    # Forge compile ONNX model
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Verify model outputs
    fw_out, co_out = verify(
        inputs,
        onnx_module,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # Print classification results
    print_cls_results(fw_out[0], co_out[0])
