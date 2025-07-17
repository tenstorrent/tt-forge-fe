# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import onnx
import paddle
import pytest
import torch
from datasets import load_dataset
from loguru import logger
from paddle.vision.models import alexnet
from tensorflow.keras.applications import ResNet50
from torchvision import transforms

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from test.models.models_utils import print_cls_results, preprocess_inputs
from test.models.tensorflow.vision.resnet.model_utils.image_utils import get_sample_inputs
from test.utils import download_model


@pytest.mark.push
@pytest.mark.sanity
@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((4, 4), torch.float32),
        ((6, 7), torch.float32),
        ((2, 3, 4), torch.float32),
    ],
)
def test_eltwise_add(shape, dtype):
    """Test element-wise addition using forge compile and verify."""

    class AddModel(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    input1 = torch.randn(shape, dtype=dtype)
    input2 = torch.randn(shape, dtype=dtype)
    inputs = [input1, input2]

    model = AddModel()
    model.eval()

    compiled_model = forge.compile(model, sample_inputs=inputs)
    verify(inputs, model, compiled_model)


@pytest.mark.nightly
def test_alexnet_torchhub():
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.ALEXNET,
        source=Source.TORCH_HUB,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load model
    framework_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "alexnet", pretrained=True).to(
        torch.bfloat16
    )
    framework_model.eval()

    # Load and pre-process image
    try:

        dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
        input_image = next(iter(dataset.skip(10)))["image"]
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        img_tensor = preprocess(input_image).unsqueeze(0)
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)

    inputs = [img_tensor.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    fw_out, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    print_cls_results(fw_out[0], co_out[0])


@pytest.mark.nightly
def test_alexnet():
    # Record model details
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.ALEXNET,
        source=Source.PADDLE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load framework model
    framework_model = alexnet(pretrained=True)

    # Compile model
    input_sample = [paddle.rand([1, 3, 224, 224])]
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=input_sample,
        module_name=module_name,
    )

    # Verify data on sample input
    verify(
        input_sample,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
    )


@pytest.mark.nightly
def test_alexnet_onnx(forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.ALEXNET,
        source=Source.TORCH_HUB,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load model
    torch_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "alexnet", pretrained=True)
    torch_model.eval()

    # Load input
    inputs = preprocess_inputs()

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/alexnet.onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification and Inference
    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # Post processing
    print_cls_results(fw_out[0], co_out[0])


@pytest.mark.push
@pytest.mark.nightly
def test_resnet_tensorflow():

    # Record model details
    module_name = record_model_properties(
        framework=Framework.TENSORFLOW,
        model=ModelArch.RESNET,
        variant="resnet50",
        source=Source.KERAS,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load Resnet50 Model
    framework_model = ResNet50(weights="imagenet")

    # Load sample inputs
    sample_input = get_sample_inputs()
    inputs = [sample_input]

    # Compile model
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify data on sample input
    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.98)),
    )
