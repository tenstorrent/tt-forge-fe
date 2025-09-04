# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import forge
import forge.op
from forge import ForgeModule

from loguru import logger
import torch

from forge import Tensor, compile
from forge.verify.verify import verify
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.config import VerifyConfig
from forge.forge_property_utils import (
    record_forge_op_name,
    record_op_model_names,
    record_forge_op_args,
    record_single_op_operands_info,
)
import pytest


class Log0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, log_input_0):
        log_output_1 = forge.op.Log("", log_input_0)
        return log_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Log0, [((1, 10), torch.float32)], {"model_names": ["onnx_mnist_base_img_cls_github"], "pcc": 0.99}),
    (Log0, [((1, 32, 480, 640), torch.float32)], {"model_names": ["onnx_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Log0, [((1, 64, 240, 320), torch.float32)], {"model_names": ["onnx_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Log0, [((1, 32, 240, 320), torch.float32)], {"model_names": ["onnx_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (
        Log0,
        [((1, 128, 120, 160), torch.float32)],
        {"model_names": ["onnx_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (Log0, [((1, 64, 120, 160), torch.float32)], {"model_names": ["onnx_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Log0, [((1, 256, 60, 80), torch.float32)], {"model_names": ["onnx_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Log0, [((1, 128, 60, 80), torch.float32)], {"model_names": ["onnx_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Log0, [((1, 512, 30, 40), torch.float32)], {"model_names": ["onnx_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Log0, [((1, 256, 30, 40), torch.float32)], {"model_names": ["onnx_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Log0, [((1, 1024, 15, 20), torch.float32)], {"model_names": ["onnx_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Log0, [((1, 512, 15, 20), torch.float32)], {"model_names": ["onnx_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
@pytest.mark.parametrize("training_test", [False, True], ids=["inference", "training"])
def test_module(forge_module_and_shapes_dtypes, training_test):

    record_forge_op_name("Log")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.get("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name in ["pcc"]:
            continue
        elif metadata_name == "model_names":
            record_op_model_names(metadata_value)
        elif metadata_name == "args":
            record_forge_op_args(metadata_value)
        else:
            logger.warning(
                "No utility function available in forge property handler to record %s property", metadata_name
            )

    max_int = 1000
    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype, max_int=max_int, requires_grad=training_test)
        for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)

    for name, parameter in framework_model._parameters.items():
        parameter_tensor = Tensor.create_torch_tensor(
            shape=parameter.shape.get_pytorch_shape(),
            dtype=parameter.pt_data_format,
            max_int=max_int,
            requires_grad=training_test,
        )
        framework_model.set_parameter(name, parameter_tensor)

    for name, constant in framework_model._constants.items():
        constant_tensor = Tensor.create_torch_tensor(
            shape=constant.shape.get_pytorch_shape(),
            dtype=constant.pt_data_format,
            max_int=max_int,
            requires_grad=training_test,
        )
        framework_model.set_constant(name, constant_tensor)

    record_single_op_operands_info(framework_model, inputs)

    compiler_cfg = forge.config.CompilerConfig()
    if "default_df_override" in metadata.keys():
        compiler_cfg.default_df_override = forge.DataFormat.from_json(metadata["default_df_override"])

    compiled_model = compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg, training=training_test)

    verify(
        inputs,
        framework_model,
        compiled_model,
        with_backward=training_test,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )
