# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import forge
import forge.op
from forge import ForgeModule

from loguru import logger
import torch

from forge import Tensor, compile
from forge.verify.compare import compare_with_golden
from forge.verify.verify import verify
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.config import VerifyConfig
import pytest


class Cumsum0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cumsum_input_0):
        cumsum_output_1 = forge.op.CumSum("", cumsum_input_0, dim=1)
        return cumsum_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    pytest.param(
        (
            Cumsum0,
            [((1, 256), torch.int64)],
            {
                "model_name": [
                    "pt_opt_facebook_opt_1_3b_clm_hf",
                    "pt_opt_facebook_opt_125m_clm_hf",
                    "pt_opt_facebook_opt_350m_clm_hf",
                ],
                "pcc": 0.99,
                "op_params": {"dim": "1"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32"
            )
        ],
    ),
    pytest.param(
        (
            Cumsum0,
            [((1, 32), torch.int64)],
            {
                "model_name": [
                    "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                    "pt_opt_facebook_opt_1_3b_qa_hf",
                    "pt_opt_facebook_opt_350m_qa_hf",
                    "pt_opt_facebook_opt_125m_seq_cls_hf",
                    "pt_opt_facebook_opt_350m_seq_cls_hf",
                    "pt_opt_facebook_opt_125m_qa_hf",
                ],
                "pcc": 0.99,
                "op_params": {"dim": "1"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32"
            )
        ],
    ),
    pytest.param(
        (
            Cumsum0,
            [((1, 128), torch.int32)],
            {
                "model_name": [
                    "pt_roberta_xlm_roberta_base_mlm_hf",
                    "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                ],
                "pcc": 0.99,
                "op_params": {"dim": "1"},
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, record_forge_property):
    record_forge_property("op_name", "CumSum")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        record_forge_property(metadata_name, metadata_value)

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

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
