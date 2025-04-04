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
import pytest


class Max0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("max0_const_1", shape=(1,), dtype=torch.int32)

    def forward(self, max_input_0):
        max_output_1 = forge.op.Max("", max_input_0, self.get_constant("max0_const_1"))
        return max_output_1


class Max1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("max1_const_1", shape=(1,), dtype=torch.float32)

    def forward(self, max_input_0):
        max_output_1 = forge.op.Max("", max_input_0, self.get_constant("max1_const_1"))
        return max_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    pytest.param(
        (
            Max0,
            [((2441216,), torch.int32)],
            {"model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Max1,
            [((1, 32, 32, 32), torch.float32)],
            {"model_name": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"], "pcc": 0.99},
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp:102: tt::exception info: ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast"
            )
        ],
    ),
    pytest.param(
        (
            Max1,
            [((1, 16, 32, 32), torch.float32)],
            {"model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp:102: tt::exception info: ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast"
            )
        ],
    ),
    pytest.param(
        (Max1, [((1, 32, 256, 256), torch.float32)], {"model_name": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99}),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp:102: tt::exception info: ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast"
            )
        ],
    ),
    pytest.param(
        (
            Max1,
            [((1, 16, 256, 256), torch.float32)],
            {
                "model_name": [
                    "pt_opt_facebook_opt_350m_clm_hf",
                    "pt_xglm_facebook_xglm_1_7b_clm_hf",
                    "pt_xglm_facebook_xglm_564m_clm_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp:102: tt::exception info: ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast"
            )
        ],
    ),
    pytest.param(
        (
            Max1,
            [((1, 12, 32, 32), torch.float32)],
            {"model_name": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp:102: tt::exception info: ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast"
            )
        ],
    ),
    pytest.param(
        (Max1, [((1, 12, 256, 256), torch.float32)], {"model_name": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99}),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp:102: tt::exception info: ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast"
            )
        ],
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):
    forge_property_recorder("tags.op_name", "Max")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        forge_property_recorder("tags." + str(metadata_name), metadata_value)

    max_int = 1000
    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype, max_int=max_int)
        for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)
    framework_model.process_framework_parameters()

    for name, parameter in framework_model._parameters.items():
        parameter_tensor = Tensor.create_torch_tensor(
            shape=parameter.shape.get_pytorch_shape(), dtype=parameter.pt_data_format, max_int=max_int
        )
        framework_model.set_parameter(name, parameter_tensor)

    for name, constant in framework_model._constants.items():
        constant_tensor = Tensor.create_torch_tensor(
            shape=constant.shape.get_pytorch_shape(), dtype=constant.pt_data_format, max_int=max_int
        )
        framework_model.set_constant(name, constant_tensor)

    compiled_model = compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
        forge_property_handler=forge_property_recorder,
    )
