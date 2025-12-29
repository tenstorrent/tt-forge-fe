# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.vision.wideresnet.model_utils.utils import (
    post_processing,
    generate_model_wideresnet_imgcls_pytorch,
)
import onnx


variants = [
    pytest.param("wide_resnet50_2", marks=pytest.mark.pr_models_regression, id="wide_resnet50_2"),
    "wide_resnet101_2",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_wideresnet_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.WIDERESNET,
        source=Source.TIMM,
        variant=variant,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load Model and input
    torch_model, inputs = generate_model_wideresnet_imgcls_pytorch(variant)

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/{variant.replace('.', '_')}.onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # Post processing
    post_processing(co_out)
