# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.onnx
import onnx

import forge
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties

from third_party.tt_forge_models.googlenet.pytorch import ModelLoader, ModelVariant


@pytest.mark.parametrize(
    "variant",
    [
        ModelVariant.GOOGLENET,
    ],
)
@pytest.mark.nightly
@pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2834")
def test_googlenet_onnx_export_from_pytorch(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.GOOGLENET,
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load model and input
    loader = ModelLoader(variant=variant)
    torch_model = loader.load_model(dtype_override=torch.float32)
    input_tensor = loader.load_inputs(dtype_override=torch.float32)
    sample_inputs = [input_tensor]

    # Export to ONNX
    onnx_path = f"{forge_tmp_path}/googlenet_{variant.name.lower()}.onnx"
    torch.onnx.export(
        torch_model,
        input_tensor,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Compile model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=sample_inputs,
        module_name=module_name,
    )

    # Verify model
    _, co_out = verify(sample_inputs, framework_model, compiled_model)

    # Print classification results
    loader.print_cls_results(co_out)
