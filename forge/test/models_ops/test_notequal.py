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


class Notequal0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("notequal0_const_1", shape=(1,), dtype=torch.int64)

    def forward(self, notequal_input_0):
        notequal_output_1 = forge.op.NotEqual("", notequal_input_0, self.get_constant("notequal0_const_1"))
        return notequal_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    pytest.param(
        (Notequal0, [((1, 11), torch.int64)], {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99}),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/broadcast_height_and_width_multi_core_program_factory.cpp:27: tt::exception info: BinaryOpType cannot be mapped to BcastOpMath"
            )
        ],
    ),
    pytest.param(
        (Notequal0, [((1, 9), torch.int64)], {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99}),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/broadcast_height_and_width_multi_core_program_factory.cpp:27: tt::exception info: BinaryOpType cannot be mapped to BcastOpMath"
            )
        ],
    ),
    pytest.param(
        (
            Notequal0,
            [((1, 128), torch.int64)],
            {
                "model_names": [
                    "pt_roberta_xlm_roberta_base_mlm_hf",
                    "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/broadcast_height_and_width_multi_core_program_factory.cpp:27: tt::exception info: BinaryOpType cannot be mapped to BcastOpMath"
            )
        ],
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):

    forge_property_recorder.enable_single_op_details_recording()
    forge_property_recorder.record_forge_op_name("NotEqual")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name == "model_names":
            forge_property_recorder.record_op_model_names(metadata_value)
        elif metadata_name == "args":
            forge_property_recorder.record_forge_op_args(metadata_value)
        else:
            logger.warning(
                "No utility function available in forge property handler to record %s property", metadata_name
            )

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

    forge_property_recorder.record_single_op_operands_info(framework_model, inputs)

    compiled_model = compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
        forge_property_handler=forge_property_recorder,
    )
