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


class TestTensorsUtils:

    dev_data_format_to_dtype = {
        # forge.DataFormat.Bfp2: torch.float16,
        forge.DataFormat.Bfp2: torch.bool,  # mapped
        forge.DataFormat.Bfp2_b: torch.bfloat16,
        forge.DataFormat.Bfp4: torch.float16,
        forge.DataFormat.Bfp4_b: torch.bfloat16,
        forge.DataFormat.Bfp8: torch.float16,
        forge.DataFormat.Bfp8_b: torch.bfloat16,
        forge.DataFormat.Float16: torch.float16,  # mapped
        forge.DataFormat.Float16_b: torch.bfloat16,  # mapped
        forge.DataFormat.Float32: torch.float32,  # mapped
        forge.DataFormat.Int8: torch.int8,  # mapped
        forge.DataFormat.Lf8: torch.float16,
        forge.DataFormat.RawUInt16: torch.uint8,
        forge.DataFormat.RawUInt32: torch.uint8,
        forge.DataFormat.RawUInt8: torch.uint8,
        forge.DataFormat.UInt16: torch.uint8,
        forge.DataFormat.Int32: torch.int32,  # mapped
    }

    @classmethod
    def get_dtype_for_df(cls, dev_data_format: forge.DataFormat = None) -> torch.dtype:

        dtype: torch.dtype

        if dev_data_format is None:
            dtype = None
        else:
            # dtype = torch.float32
            if dev_data_format in cls.dev_data_format_to_dtype:
                dtype = cls.dev_data_format_to_dtype[dev_data_format]
            else:
                raise ValueError(f"Unsupported dtype for dev data format: {dev_data_format}")

            if dtype in (torch.uint8,):
                dtype = torch.int8

        return dtype

    @classmethod
    def move_value_range(cls, inputs: List[torch.Tensor], min, max) -> List[torch.Tensor]:
        inputs = [min + (max - min) * input for input in inputs]
        return inputs

    @classmethod
    def get_random_sign(cls, generator: torch.Generator) -> int:
        return 2 * torch.randint(0, 2, (1,), generator=generator).item() - 1

    @classmethod
    def get_random_torch_inputs(
        cls, dtype: torch.dtype, input_shapes: List[TensorShape], generator: torch.Generator
    ) -> List[torch.Tensor]:

        if dtype is None:
            inputs = [torch.rand(input_shape, generator=generator) for input_shape in input_shapes]
        elif dtype in (torch.float16, torch.bfloat16, torch.float32):
            # TODO Testing big and small numbers should be configurable depending on operator
            # Some operators support mix small and big numbers
            # TODO Testing small numbers close to 0
            inputs = [torch.rand(input_shape, dtype=dtype, generator=generator) for input_shape in input_shapes]
            min, max = 100, 10000
            sign = cls.get_random_sign(generator)
            min, max = sign * min, sign * max
            inputs = cls.move_value_range(inputs, min, max)
        elif dtype in (torch.bool,):
            inputs = [
                torch.rand(input_shape, dtype=torch.float32, generator=generator) > 0.5 for input_shape in input_shapes
            ]
        elif dtype in (torch.uint8,):
            inputs = [
                torch.randint(low=0, high=256, size=input_shape, dtype=dtype, generator=generator)
                for input_shape in input_shapes
            ]
        elif dtype in (torch.int8,):
            inputs = [
                torch.randint(low=-128, high=127, size=input_shape, dtype=dtype, generator=generator)
                for input_shape in input_shapes
            ]
        else:
            raise ValueError(f"Fail creating random torch input for unsupported dtype: {dtype}")

        return inputs


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

    dtype = TestTensorsUtils.get_dtype_for_df(dev_data_format)

    # if dtype is not None:
    #     torch.set_default_dtype(dtype)

    # forge.config.set_configuration_options(default_df_override=dev_data_format)

    inputs = TestTensorsUtils.get_random_torch_inputs(dtype, input_shapes, generator)

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
