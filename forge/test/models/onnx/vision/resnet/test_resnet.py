# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import random
import onnx
import torch
from datasets import load_dataset
from transformers import ResNetForImageClassification

import forge
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker

from test.models.utils import Framework, Source, Task, build_module_name


variants = [
    "microsoft/resnet-50",
]

# Opset 7 is the minimum version in Torch.
# Opset 17 is the maximum version in Torchscript.
opset_versions = [7, 17]


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.parametrize("opset_version", opset_versions, ids=opset_versions)
def test_resnet_onnx(forge_property_recorder, variant, tmp_path, opset_version):
    random.seed(0)

    # Record model details
    module_name = build_module_name(
        framework=Framework.ONNX,
        model="resnet",
        variant="50",
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )
    forge_property_recorder.record_model_name(module_name)

    # Export model to ONNX
    torch_model = ResNetForImageClassification.from_pretrained(variant)
    input_sample = torch.randn(1, 3, 224, 224)
    onnx_path = f"{tmp_path}/resnet50.onnx"
    torch.onnx.export(torch_model, input_sample, onnx_path, opset_version=opset_version)

    # Load framework model
    # TODO: Replace with pre-generated ONNX model to avoid exporting from scratch.
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    input_sample = [input_sample]
    compiled_model = forge.compile(onnx_model, input_sample, forge_property_handler=forge_property_recorder)

    # Verify data on sample input
    verify(
        input_sample,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
        forge_property_handler=forge_property_recorder,
    )
