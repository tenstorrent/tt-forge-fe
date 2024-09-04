# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import urllib

import onnx
import pytest
import numpy as np
import onnxruntime
import torch
import forge
from forge import (
    OnnxModule,
    VerifyConfig,
    DataFormat,
    BackendDevice,
)
from forge.verify import verify_module
from forge.verify.config import TestKind
from forge.config import _get_global_compiler_config




def test_onnx_quantized_resnet(test_device):
    # Skip test on blackhole until we have support for quantized models on blackhole forge#2700
    if test_device.arch == BackendDevice.Blackhole:
        pytest.skip("Blackhole does not support quantized models")

    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip()

    # Download ONNX model
    save_path = "third_party/confidential_customer_models/quantized/ResNet50-v1.5-Int8.onnx"
    if not os.path.exists(save_path):
        raise RuntimeError("Model not found")

    # LOAD ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    forge_onnx_model = OnnxModule(
        "onnx_quantized_ResNet50",
        onnx_model,
        save_path,
    )
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.graph_solver_self_cut_type = "FastCut"
    compiler_cfg.default_df_override = DataFormat.Float32

    # os.environ["FORGE_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["FORGE_FRACTURIZATION_DISABLE"] = "1"
    # os.environ["FORGE_REPRODUCE_SUBGRAPH"] = "1"
    # os.environ["FORGE_REPRODUCE_SUBGRAPH_INPUT"] = "quantize_0.dc.forge_quantize.1"
    # os.environ["FORGE_REPRODUCE_SUBGRAPH_OUTPUT"] = "conv2d_1.dc.matmul.11"

    # Sanity run
    input_shape = []
    for i in range(len(onnx_model.graph.input)):
        dimension = onnx_model.graph.input[i].type.tensor_type.shape
        i_shape = [d.dim_value for d in dimension.dim]
        input_shape.append(i_shape)
    
    # tti_path = "onnx_int8_resnet50_epoch_0.tti"
    # if not os.path.exists(tti_path):
    #     tt_module = forge_onnx_model
    #     device = forge.TTDevice(
    #         "tt0", module=tt_module,arch=forge.BackendDevice.Wormhole_B0, devtype=forge.BackendType.Silicon)
    #     tti_img = device.compile_to_image(
    #         img_path=tti_path,
    #         training=False,
    #         sample_inputs=[torch.randn(shape) for shape in input_shape],
    #     )


    # device_img: forge.TTDeviceImage = forge.TTDeviceImage.load_from_disk(tti_path)
    # ttdevice = forge.TTDevice.load_image(img=device_img)

    # inputs = [torch.randn(shape) for shape in input_shape]
    # ttdevice.push_to_inputs(*inputs)
    # output_q = forge.run_inference(_sequential=True)
    # output = output_q.get()[0].value().detach()

    # golden_output = forge_onnx_model.forward(*inputs)
    # assert np.allclose(output, golden_output[0], atol=1e-3, rtol=1e-3)
    # Compile and verify
    verify_module(
        forge_onnx_model,
        input_shape,
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            # verify_forge_codegen_vs_framework=True,
            verify_all=True,
        ),
    )

