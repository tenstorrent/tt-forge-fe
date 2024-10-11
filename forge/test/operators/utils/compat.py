# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Compatibility methods and datatypes for Forge

import forge
import torch

from loguru import logger
from typing import Optional, List

from forge import ForgeModule, Module, VerifyConfig
from forge.op_repo import TensorShape
from forge.op.eval.common import compare_with_golden


# TODO - Remove this class once TestDevice is available in Forge
# https://github.com/tenstorrent/tt-forge-fe/issues/342
class TestDevice:

    pass


# Compatibility method for verifying models
def verify_module(
    model: Module,
    input_shapes: List[TensorShape],
    pcc: Optional[float] = None,
    dev_data_format: forge.DataFormat = None,
):

    logger.debug(
        f"Verifying model class: {model.__class__.__name__}({model.__class__.__base__.__module__}.{model.__class__.__base__.__name__}) input_shapes: {input_shapes}"
    )

    # TODO configure manual seed
    generator = torch.Generator().manual_seed(42)

    # forge.config.set_configuration_options(default_df_override=dev_data_format)

    inputs = [torch.rand(input_shape, generator=generator) for input_shape in input_shapes]

    fw_out = model(*inputs)

    forge_inputs = [forge.Tensor.create_from_torch(input, dev_data_format=dev_data_format) for input in inputs]

    compiled_model = forge.compile(model, sample_inputs=forge_inputs)
    co_out = compiled_model(*forge_inputs)

    # TODO check output data format type

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    # It would be good that compare_with_golden_pcc can take pcc as None
    if pcc is not None:
        assert all(
            [compare_with_golden(golden=fo, calculated=co, pcc=pcc) for fo, co in zip(fw_out, co_out)]
        ), "PCC check failed"
    else:
        assert all(
            [compare_with_golden(golden=fo, calculated=co) for fo, co in zip(fw_out, co_out)]
        ), "PCC check failed"
