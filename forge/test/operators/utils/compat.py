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

from .datatypes import OperatorParameterTypes, ValueRanges


# TODO - Remove this class once TestDevice is available in Forge
# https://github.com/tenstorrent/tt-forge-fe/issues/342
class TestDevice:

    pass


class TestTensorsUtils:

    dev_data_format_to_dtype = {
        # forge.DataFormat.Bfp2: torch.float16,
        # forge.DataFormat.Bfp2: torch.bfloat16,
        forge.DataFormat.Bfp2: torch.float32,
        forge.DataFormat.Bfp2_b: torch.float32,
        # forge.DataFormat.Bfp2_b: torch.bool,  # mapped
        forge.DataFormat.Bfp4: torch.float32,
        forge.DataFormat.Bfp4_b: torch.float32,
        forge.DataFormat.Bfp8: torch.float32,
        forge.DataFormat.Bfp8_b: torch.float32,
        # forge.DataFormat.Float16: torch.float16,  # fatal python error
        forge.DataFormat.Float16: torch.float32,  # mapped
        forge.DataFormat.Float16_b: torch.float32,  # mapped
        forge.DataFormat.Float32: torch.float32,  # mapped
        # forge.DataFormat.Lf8: torch.float16,  # fatal python error
        # forge.DataFormat.Lf8: torch.bfloat16,  # pcc failed
        forge.DataFormat.Lf8: torch.float32,
        # forge.DataFormat.RawUInt8: torch.int8,  # unsupported
        # forge.DataFormat.RawUInt8: torch.uint8,  # Unsupported torch dtype
        forge.DataFormat.RawUInt8: torch.int32,
        # forge.DataFormat.RawUInt16: torch.int8,
        # forge.DataFormat.RawUInt16: torch.uint8,
        # forge.DataFormat.RawUInt16: torch.int16,
        forge.DataFormat.RawUInt16: torch.int32,
        # forge.DataFormat.RawUInt32: torch.uint8,
        forge.DataFormat.RawUInt32: torch.int32,
        # forge.DataFormat.Int8: torch.int8,  # unsupported
        forge.DataFormat.Int8: torch.int32,  # mapped
        # forge.DataFormat.UInt16: torch.int8,  #  E  RuntimeError: Unsupported data type
        forge.DataFormat.UInt16: torch.int32,
        forge.DataFormat.Int32: torch.int32,  # mapped
    }

    data_format_ranges = {
        forge.DataFormat.Bfp2: (-10000, 10000),
        forge.DataFormat.Bfp2_b: (-10000, 10000),
        forge.DataFormat.Bfp4: (-10000, 10000),
        forge.DataFormat.Bfp4_b: (-10000, 10000),
        forge.DataFormat.Bfp8: (-10000, 10000),
        forge.DataFormat.Bfp8_b: (-10000, 10000),
        forge.DataFormat.Float16: (-10000, 10000),
        forge.DataFormat.Float16_b: (-10000, 10000),
        forge.DataFormat.Float32: (-10000, 10000),
        forge.DataFormat.Lf8: (-10000, 10000),
        forge.DataFormat.RawUInt8: (0, 2**8 - 1),
        forge.DataFormat.RawUInt16: (0, 2**16 - 1),
        forge.DataFormat.RawUInt32: (0, 2**32 - 1),
        forge.DataFormat.Int8: (-(2**7), 2**7 - 1),
        forge.DataFormat.UInt16: (0, 2**16 - 1),
        forge.DataFormat.Int32: (-(2**31), 2**31 - 1),
    }

    dtype_ranges = {
        torch.bfloat16: (-10000, 10000),
        torch.float16: (-10000, 10000),
        torch.float32: (-10000, 10000),
        torch.uint8: (0, 2**8 - 1),
        torch.int8: (-(2**7), 2**7 - 1),
        torch.int16: (-(2**15), 2**15 - 1),
        torch.int32: (-(2**31), 2**31 - 1),
    }

    @classmethod
    def get_value_range(
        cls,
        dev_data_format: forge.DataFormat,
        dtype: torch.dtype,
        value_range: Optional[OperatorParameterTypes.RangeValue] = None,
    ) -> OperatorParameterTypes.RangeValue:
        if dev_data_format in cls.data_format_ranges:
            data_format_ranges = cls.data_format_ranges[dev_data_format]
        else:
            raise ValueError(f"Unsupported range for dev data format: {dev_data_format}")
        if dtype in cls.dtype_ranges:
            dtype_ranges = cls.dtype_ranges[dtype]
        else:
            raise ValueError(f"Unsupported range for dtype: {dtype}")
        range = cls.intersect_ranges(data_format_ranges, dtype_ranges)
        if value_range is not None:
            range = cls.intersect_ranges(range, value_range)
        return range

    @classmethod
    def intersect_ranges(
        cls, range1: OperatorParameterTypes.RangeValue, range2: OperatorParameterTypes.RangeValue
    ) -> OperatorParameterTypes.RangeValue:

        low = None
        if range1[0] is None and range2[0] is None:
            low = None
        else:
            if range1[0] is None:
                low = range2[0]
            elif range2[0] is None:
                low = range1[0]
            else:
                low = max(range1[0], range2[0])

        high = None
        if range1[1] is None and range2[1] is None:
            high = None
        else:
            if range1[1] is None:
                high = range2[1]
            elif range2[1] is None:
                high = range1[1]
            else:
                high = min(range1[1], range2[1])

        return low, high

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

        return dtype

    @classmethod
    def move_from_small_to_big_value_range(cls, input: torch.Tensor, min, max) -> torch.Tensor:
        return min + (max - min) * input

    @classmethod
    def get_random_sign(cls, generator: torch.Generator) -> int:
        return 2 * torch.randint(0, 2, (1,), generator=generator).item() - 1

    @classmethod
    def get_random_torch_input(
        cls,
        dev_data_format: forge.DataFormat,
        dtype: torch.dtype,
        input_shape: TensorShape,
        generator: torch.Generator,
        value_range: Optional[OperatorParameterTypes.RangeValue] = None,
    ) -> List[torch.Tensor]:

        if dtype is None:
            return torch.rand(input_shape, generator=generator)
        elif dtype in (torch.float16, torch.bfloat16, torch.float32):
            input = torch.rand(input_shape, dtype=dtype, generator=generator)
            range = cls.get_value_range(dev_data_format=dev_data_format, dtype=dtype, value_range=value_range)
            if range[0] >= 0:
                # only positive values
                sign = 1
                min, max = range
            elif range[1] <= 0:
                # only negative values
                sign = -1
                max, min = range
                min, max = abs(min), abs(max)
            else:
                # positive and negative values
                sign = cls.get_random_sign(generator)
                if sign == 1:
                    min, max = 0, range[1]
                else:
                    min, max = 0, range[0]
            min, max = sign * min, sign * max
            return cls.move_from_small_to_big_value_range(input, min, max)
        elif dtype in (torch.bool,):
            return torch.rand(input_shape, dtype=torch.float32, generator=generator) > 0.5
        elif dtype in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
        ):
            range = cls.get_value_range(dev_data_format=dev_data_format, dtype=dtype, value_range=value_range)
            low, high = range
            return torch.randint(low=low, high=high, size=input_shape, dtype=dtype, generator=generator)
        else:
            raise ValueError(f"Fail creating random torch input for unsupported dtype: {dtype}")


# Compatibility method for verifying models
def verify_module(
    model: Module,
    input_shapes: List[TensorShape],
    pcc: Optional[float] = None,
    dev_data_format: forge.DataFormat = None,
    value_range: Optional[ValueRanges] = None,
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

    value_range = value_range.value.get_range() if value_range is not None else None
    inputs = [
        TestTensorsUtils.get_random_torch_input(dev_data_format, dtype, input_shape, generator, value_range)
        for input_shape in input_shapes
    ]

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
