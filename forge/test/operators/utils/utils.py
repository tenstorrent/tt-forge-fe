# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Operator test utilities

import random
import sys
import forge
import torch
import pytest


from enum import Enum
from dataclasses import dataclass
from loguru import logger
from typing import Optional, List, Dict, Type, Union, Tuple

from forge import ForgeModule, Module, VerifyConfig
from forge.op_repo import TensorShape
from forge.verify import TestKind  #, verify_module
from forge.config import _get_global_compiler_config
from forge._C import MathFidelity

from .compat import TestDevice, verify_module


# All supported framework model types
FrameworkModelType = Union [
    Type[torch.nn.Module],
]


class ShapeUtils:

    def get_basic_shapes(*shape_dim: int, microbatch: bool = None) -> List[Tuple]:
        '''
        Get basic shapes for single operator testing.
        '''

         # 2-dimensional shape, microbatch_size = 1:
        dim2_no_microbatch = [
            (1, 4),                 #00      # 3.1 Full tensor (i.e. full expected shape)
            (1, 17),                #01      # 3.1 Full tensor (i.e. full expected shape)
            (1, 23),                #02      # 3.2 Tensor reduce on one or more dims to 1
            (1, 1),                 #03      # 3.2 Tensor reduce on one or more dims to 1
            (1, 100),               #04      # 4.3 Very large (thousands, 10s of thousands)
            (1, 500),               #05      # 4.3 Very large (thousands, 10s of thousands)
            (1, 1000),              #06      # 4.4 Extreme ratios between height/width
            (1, 1920),              #07      # 4.4 Extreme ratios between height/width
            (1, 10000),             #08      # 4.4 Extreme ratios between height/width
            (1, 64),                #09      # 4.1 Divisible by 32
            (1, 96),                #10      # 4.1 Divisible by 32
            (1, 41),                #11      # 4.2 Prime numbers
            (1, 3),                 #12      # 4.2 Prime numbers
        ]

        # 2-dimensional shape, microbatch_size > 1:
        dim2_microbatch = [
            (3, 4),                 #13      # 3.1 Full tensor (i.e. full expected shape)
            (45, 17),               #14      # 3.1 Full tensor (i.e. full expected shape)
            (64, 1),                #15      # 3.2 Tensor reduce on one or more dims to 1
            (100, 100),             #16      # 4.3 Very large (thousands, 10s of thousands)
            (1000, 100),            #17      # 4.3 Very large (thousands, 10s of thousands)
            (10, 1000),             #18      # 4.4 Extreme ratios between height/width
            (9920, 1),              #19      # 4.4 Extreme ratios between height/width
            (10000, 1),             #20      # 4.4 Extreme ratios between height/width
            (32, 64),               #21      # 4.1 Divisible by 32
            (160, 96),              #22      # 4.1 Divisible by 32
            (17, 41),               #23      # 4.2 Prime numbers
            (89, 3),                #24      # 4.2 Prime numbers
        ]

        # 3-dimensional shape, microbatch_size = 1:
        dim3_no_microbatch = [
            (1, 3, 4),              #25     # 3.1 Full tensor (i.e. full expected shape)
            (1, 45, 17),            #26     # 3.1 Full tensor (i.e. full expected shape)
            (1, 1, 23),             #27     # 3.2 Tensor reduce on one or more dims to 1
            (1, 64, 1),             #28     # 3.2 Tensor reduce on one or more dims to 1
            (1, 100, 100),          #29     # 4.3 Very large (thousands, 10s of thousands)
            (1, 1000, 100),         #30     # 4.3 Very large (thousands, 10s of thousands)
            (1, 10, 1000),          #31     # 4.4 Extreme ratios between height/width
            (1, 9920, 1),           #32     # 4.4 Extreme ratios between height/width
            (1, 10000, 1),          #33     # 4.4 Extreme ratios between height/width
            (1, 32, 64),            #34     # 4.1 Divisible by 32
            (1, 160, 96),           #35     # 4.1 Divisible by 32
            (1, 17, 41),            #36     # 4.2 Prime numbers
            (1, 89, 3),             #37     # 4.2 Prime numbers
        ]

        # 3-dimensional shape, microbatch_size > 1:
        dim3_microbatch = [
            (2, 3, 4),              #38     # 3.1 Full tensor (i.e. full expected shape)
            (11, 45, 17),           #39     # 3.1 Full tensor (i.e. full expected shape)
            (11, 1, 23),            #40     # 3.2 Tensor reduce on one or more dims to 1
            (11, 64, 1),            #41     # 3.2 Tensor reduce on one or more dims to 1
            (100, 100, 100),        #42     # 4.3 Very large (thousands, 10s of thousands)
            (10, 1000, 100),        #43     # 4.3 Very large (thousands, 10s of thousands)
            (10, 10000, 1),         #44     # 4.4 Extreme ratios between height/width
            (32, 32, 64),           #45     # 4.1 Divisible by 32
            (64, 160, 96),          #46     # 4.1 Divisible by 32
            (11, 17, 41),           #47     # 4.2 Prime numbers
            (13, 89, 3),            #48     # 4.2 Prime numbers
        ]

        # 4-dimensional shape, microbatch_size = 1:
        dim4_no_microbatch = [
            (1, 2, 3, 4),           #49     # 3.1 Full tensor (i.e. full expected shape)
            (1, 11, 45, 17),        #50     # 3.1 Full tensor (i.e. full expected shape)
            (1, 11, 1, 23),         #51     # 3.2 Tensor reduce on one or more dims to 1
            (1, 11, 64, 1),         #52     # 3.2 Tensor reduce on one or more dims to 1
            (1, 100, 100, 100),     #53     # 4.3 Very large (thousands, 10s of thousands)
            (1, 10, 1000, 100),     #54     # 4.3 Very large (thousands, 10s of thousands)
            (1, 1, 10, 1000),       #55     # 4.4 Extreme ratios between height/width
            (1, 1, 9920, 1),        #56     # 4.4 Extreme ratios between height/width
            (1, 10, 10000, 1),      #57     # 4.4 Extreme ratios between height/width
            (1, 32, 32, 64),        #58     # 4.1 Divisible by 32
            (1, 64, 160, 96),       #59     # 4.1 Divisible by 32
            (1, 11, 17, 41),        #60     # 4.2 Prime numbers
            (1, 13, 89, 3),         #61     # 4.2 Prime numbers
        ]

        # 4-dimensional shape, microbatch_size > 1:
        dim4_microbatch = [
            (3, 11, 45, 17),        #62     # 3.1 Full tensor (i.e. full expected shape)
            (2, 2, 3, 4),           #63     # 3.1 Full tensor (i.e. full expected shape)
            (4, 11, 1, 23),         #64     # 3.2 Tensor reduce on one or more dims to 1
            (5, 11, 64, 1),         #65     # 3.2 Tensor reduce on one or more dims to 1
            (6, 100, 100, 100),     #66     # 4.3 Very large (thousands, 10s of thousands)
            (7, 10, 1000, 100),     #67     # 4.3 Very large (thousands, 10s of thousands)
            (8, 1, 10, 1000),       #68     # 4.4 Extreme ratios between height/width
            (9, 1, 9920, 1),        #69     # 4.4 Extreme ratios between height/width
            (10, 10, 10000, 1),     #70     # 4.4 Extreme ratios between height/width
            (11, 32, 32, 64),       #71     # 4.1 Divisible by 32
            (12, 64, 160, 96),      #72     # 4.1 Divisible by 32
            (13, 11, 17, 41),       #73     # 4.2 Prime numbers
            (14, 13, 89, 3),        #74     # 4.2 Prime numbers
        ]

        result = []

        if 2 in shape_dim:
            if microbatch is False:
                result = [*result, *dim2_no_microbatch]
            elif microbatch is True:
                result = [*result, *dim2_microbatch]
            else:
                result = [*result, *dim2_no_microbatch, *dim2_microbatch]
        if 3 in shape_dim:
            if microbatch is False:
                result = [*result, *dim3_no_microbatch]
            elif microbatch is True:
                result = [*result, *dim3_microbatch]
            else:
                result = [*result, *dim3_no_microbatch, *dim3_microbatch]
        if 4 in shape_dim:
            if microbatch is False:
                result = [*result, *dim4_no_microbatch]
            elif microbatch is True:
                result = [*result, *dim4_microbatch]
            else:
                result = [*result, *dim4_no_microbatch, *dim4_microbatch]

        return result

    @staticmethod
    def get_shape_params(*shape_dim: int, microbatch: bool = None) -> List[pytest.param]:
        '''
        Extend basic shapes to return as pytest parameters.
        Ids are set as shape=shape.
        '''
        return ShapeUtils.create_pytest_params(ShapeUtils.get_basic_shapes(*shape_dim, microbatch=microbatch), id_name="shape")

    @staticmethod
    def create_pytest_params(input_list: list, id_name, mark = ()) -> list[pytest.param]:
        params = list()
        for item in input_list:
            id = item
            if type(item) not in [tuple, list, int, float, str, bool, ]:
                id = item.__name__
            params.append(pytest.param(item, id=f"{id_name}={id}", marks=mark))
        return params

    @staticmethod
    def combine_two_params_lists(input_list_1: list[pytest.param], input_list_2: list[pytest.param]) -> list[pytest.param]:
        result_list = list()
        for item_1 in input_list_1:
            for item_2 in input_list_2:
                marks = [*item_1.marks, *item_2.marks]
                result_list.append(pytest.param(*item_1.values, *item_2.values, marks=marks, id=f"{item_1.id}_{item_2.id}"))
        return result_list
    
    @staticmethod
    def alter_shape_params(params_list_to_alter: List[pytest.param], *shapes_to_extend: Tuple[Tuple, str]) -> List[pytest.param]:
        '''
        Extend specified shapes in params_list_to_alter with marks.
        '''
        parameters = params_list_to_alter.copy()
        for shape, mark in shapes_to_extend:
            for param in parameters:
                if param.values[:len(shape)] == shape:
                    index = parameters.index(param)
                    current_marks = list(param.marks)
                    current_marks.append(mark)
                    parameters.remove(param)
                    values = param.values
                    param_to_insert = pytest.param(*values, marks=current_marks, id=param.id)
                    parameters.insert(index, param_to_insert)
        return parameters

    @staticmethod
    def get_default_df_param():
        return pytest.param(forge.DataFormat.Float16_b, id="dev-data-format=Float16_b")

    @staticmethod
    def get_default_mf_param():
        return pytest.param(forge.MathFidelity.HiFi4, id="math-fidelity=HiFi4")

    @staticmethod
    def add_df_mf_params(list_to_append: List[pytest.param], values: tuple, id: str) -> List[pytest.param]:
        '''
        Add data format and math fidelity parameters to a list of params
        '''
        result = list_to_append.copy()
        result.append(pytest.param(*values, forge.DataFormat.Float16_b, forge.MathFidelity.LoFi,  id=f"{id}_dev-data-format=Float16_b_math-fidelity=LoFi"))
        result.append(pytest.param(*values, forge.DataFormat.Float16_b, forge.MathFidelity.HiFi2, id=f"{id}_dev-data-format=Float16_b_math-fidelity=HiFi2"))
        result.append(pytest.param(*values, forge.DataFormat.Float16_b, forge.MathFidelity.HiFi3, id=f"{id}_dev-data-format=Float16_b_math-fidelity=HiFi3"))
        result.append(pytest.param(*values, forge.DataFormat.Float16_b, forge.MathFidelity.HiFi4, id=f"{id}_dev-data-format=Float16_b_math-fidelity=HiFi4"))

        result.append(pytest.param(*values, forge.DataFormat.Bfp2,      forge.MathFidelity.HiFi4, id=f"{id}_dev-data-format=Bfp2_math-fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Bfp2_b,    forge.MathFidelity.HiFi4, id=f"{id}_dev-data-format=Bfp2_b_math-fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Bfp4,      forge.MathFidelity.HiFi4, id=f"{id}_dev-data-format=Bfp4_math-fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Bfp4_b,    forge.MathFidelity.HiFi4, id=f"{id}_dev-data-format=Bfp4_b_math-fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Bfp8,      forge.MathFidelity.HiFi4, id=f"{id}_dev-data-format=Bfp8_math-fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Bfp8_b,    forge.MathFidelity.HiFi4, id=f"{id}_dev-data-format=Bfp8_b_math-fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Float16,   forge.MathFidelity.HiFi4, id=f"{id}_dev-data-format=Float16_math-fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Float16_b, forge.MathFidelity.HiFi4, id=f"{id}_dev-data-format=Float16_b_math-fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Float32,   forge.MathFidelity.HiFi4, id=f"{id}_dev-data-format=Float32_math-fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Int8,      forge.MathFidelity.HiFi4, id=f"{id}_dev-data-format=Int8_math-fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Lf8,       forge.MathFidelity.HiFi4, id=f"{id}_dev-data-format=Lf8_math-fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.RawUInt16, forge.MathFidelity.HiFi4, id=f"{id}_dev-data-format=RawUInt16_math-fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.RawUInt32, forge.MathFidelity.HiFi4, id=f"{id}_dev-data-format=RawUInt32_math-fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.RawUInt8,  forge.MathFidelity.HiFi4, id=f"{id}_dev-data-format=RawUInt8_math-fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.UInt16,    forge.MathFidelity.HiFi4, id=f"{id}_dev-data-format=UInt16_math-fidelity=HiFi4"))

        return result

    @staticmethod
    def reduce_microbatch_size(shape: TensorShape) -> TensorShape:
        '''
        Reduce microbatch dimension of a shape to 1
        Usually used for calculating shape of a constant tensor
        '''
        return (1, ) + shape[1:]
    
    @staticmethod
    def extend_shapes_with_id(shapes: List[TensorShape]) -> List[TensorShape]:
        '''
        Extend shapes with an id
        '''
        shapes_with_ids = []
        for shape in shapes:
            if type(shape) is tuple:
                shapes_with_ids.append(pytest.param(shape, marks=tuple(), id=f"shape={shape}"))
            else:
                shapes_with_ids.append(pytest.param(shape.values[0], marks=shape.marks, id=f"shape={shape.values[0]}"))
        return shapes_with_ids


@dataclass(frozen=True)
class InputSourceFlag:
    '''Dataclass for specifying compiler flags for specific input source'''
    input_queues_on_host: bool
    set_default_dram_parameters: bool
    default_dram_parameters: Optional[bool]


class InputSourceFlags(Enum):
    '''Enums defining input source flags'''
    FROM_HOST = InputSourceFlag(True, False, None)
    FROM_DRAM = InputSourceFlag(False, False, None)
    FROM_DRAM_PROLOGUED = InputSourceFlag(False, True, False)
    FROM_DRAM_NOT_PROLOGUED = InputSourceFlag(False, True, True)
    FROM_DRAM_PROLOGUE_MICROBATCH_SIZE = InputSourceFlag(False, True, None)


class CompilerUtils:
    '''Utility functions for Forge compiler configuration'''

    @staticmethod
    def set_input_source(input_source_flag: InputSourceFlag):
        '''Set compiler configuration for input source'''
        compiler_cfg = _get_global_compiler_config()
        compiler_cfg.input_queues_on_host = input_source_flag.input_queues_on_host
        if input_source_flag.set_default_dram_parameters:
            compiler_cfg.default_dram_parameters = input_source_flag.default_dram_parameters

    @staticmethod
    def set_math_fidelity(math_fidelity: MathFidelity):
        '''Set compiler configuration for math fidelity'''
        compiler_cfg = _get_global_compiler_config()
        compiler_cfg.default_math_fidelity = math_fidelity


class VerifyUtils:
    '''Utility functions for Forge verification'''

    @staticmethod
    def verify(
        model: Module,
        test_device: TestDevice,
        input_shapes: List[TensorShape],
        input_params: List[Dict] = [],
        pcc: Optional[float] = None,
        input_source_flag: InputSourceFlags = None,
        dev_data_format: forge.DataFormat = None,
        math_fidelity: forge.MathFidelity = None,
        ):
        '''Perform Forge verification on the model

        Args:
            model: Forge model
            test_device: TestDevice
            input_shapes: List of input shapes
            input_params: List of input parameters
            pcc: PCC value for verification
            input_source_flag: Input source flag
            dev_data_format: Data format
            math_fidelity: Math fidelity
        '''

        if input_source_flag:
            CompilerUtils.set_input_source(input_source_flag.value)

        if math_fidelity:
            CompilerUtils.set_math_fidelity(math_fidelity)

        # if dev_data_format:
        #     input_params.append({"dev_data_format": dev_data_format})

        verify_module(
            model,
            input_shapes=input_shapes,
            # verify_cfg=VerifyConfig(
            #     test_kind=TestKind.INFERENCE,
            #     devtype=test_device.devtype,
            #     arch=test_device.arch,
            #     pcc=pcc,
            # ),
            # input_params=[input_params],
            pcc=pcc,
            dev_data_format=dev_data_format,
        )


class LoggerUtils:
    '''Utility functions for logging'''

    @staticmethod
    def set_log_level(package_name: str, level: str):
        ''' Set log level for package_name and its subpackages

        Args:
            package_name (str): package name
            level (str): log level
        '''
        logger.add(sys.stdout, level=level, filter=lambda record: record["name"].startswith(package_name))


class RateLimiter:
    '''Rate limiter class to limit the number of allowed operations by a rate limit factor'''

    def __init__(self, rng: random.Random, max_limit: int, current_limit: int):
        self.rng = rng
        self.max_limit = max_limit
        self.current_limit = current_limit
        self.current_value: int = None

    def is_allowed(self) -> bool:
        '''Check if the operation is allowed by the rate limit factor and current random value'''
        self.current_value = self.rng.randint(1, self.max_limit)
        return self.current_value <= self.current_limit

    def limit_info(self) -> str:
        '''Return the rate limit info for previous operation'''
        if self.current_value < self.current_limit:
            return f"{self.current_value} <= {self.current_limit}"
        else:
            return f"{self.current_value} > {self.current_limit}"
