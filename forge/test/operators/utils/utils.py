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

    def __init__(self):
        self.test_list = []

    @staticmethod
    def get_basic_shapes(*shape_dim: int, microbatch: bool = None) -> List[Tuple]:
        """
        Get basic shapes for single operator testing.

        This method generates a list of basic shapes (tuples of integers) that can be used for testing single operators.
        The shapes are generated based on the provided dimensions and an optional microbatch flag.

        Args:
            shape_dim (int): Variable length argument list of integers representing the dimensions for the shapes.
            microbatch (bool, optional): A flag indicating whether to include microbatch shapes. Defaults to None.

        Returns:
            List[Tuple[int, ...]]: A list of tuples, where each tuple represents a shape with the specified dimensions.
        """

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
        """
        Generate pytest parameters for given shape dimensions.
        Args:
            shape_dim (int): Variable length argument list of shape dimensions.
            microbatch (bool, optional): A flag indicating whether to include microbatch shapes. Defaults to None.
        Returns:
            List[pytest.param]: A list of pytest parameters generated from the given shape dimensions.
        """

        return ShapeUtils.create_pytest_params(ShapeUtils.get_basic_shapes(*shape_dim, microbatch=microbatch), id_name="shape")

    @staticmethod
    def create_pytest_params(input_list: list, id_name, mark = ()) -> List[pytest.param]:
        """
        Create a list of pytest parameters from the input list.

        Args:
            input_list (list): The list of items to convert into pytest parameters.
            id_name (str): The name to use for the id of each pytest parameter.
            mark (tuple, optional): Marks to apply to each pytest parameter. Defaults to ().

        Returns:
            list[pytest.param]: A list of pytest parameters.
        """

        params = []
        for item in input_list:
            id = item
            if type(item) not in [tuple, list, int, float, str, bool, ]:
                id = item.__name__
            params.append(pytest.param(item, id=f"{id_name}={id}", marks=mark))
        return params

    @staticmethod
    def join_two_params_lists(input_list_1: list[pytest.param], input_list_2: list[pytest.param]) -> List[pytest.param]:
        """
        Joins two lists of pytest parameters into a single list.
        This function takes two lists of pytest parameters and combines each element
        from the first list with each element from the second list. The resulting list
        contains pytest parameters with combined values and marks.
        Args:
            input_list_1 (list[pytest.param]): The first list of pytest parameters.
            input_list_2 (list[pytest.param]): The second list of pytest parameters.
        Returns:
            list[pytest.param]: A list of pytest parameters with combined values and marks.
        """

        result_list = []
        for item_1 in input_list_1:
            for item_2 in input_list_2:
                marks = [*item_1.marks, *item_2.marks]
                result_list.append(pytest.param(*item_1.values, *item_2.values, marks=marks, id=f"{item_1.id}-{item_2.id}"))
        return result_list
    
    def generate_test_params_list(self, *input_list: list[pytest.param]):
        """
        Generates a list of test parameters by joining multiple lists of pytest parameters.
        Args:
            *input_list (list[pytest.param]): Variable number of lists containing pytest parameters.
        Returns:
            Self: The instance of the class.
        """

        result_list = input_list[0]
        for i in range(1, len(input_list)):
            if isinstance(input_list[i], list):
                result_list = ShapeUtils.join_two_params_lists(result_list, input_list[i])
            else:
                result_list = ShapeUtils.join_two_params_lists(result_list, [input_list[i]])
        self.test_list = result_list
        return self
    
    @staticmethod
    def __contains_subtuple_with_wildcards_and_options(tuple1, tuple2, wildcard=None) -> bool:
        """
        Check if `tuple2` is a subtuple of `tuple1` with support for wildcards and options.
        Args:
            tuple1 (tuple): The main tuple to search within.
            tuple2 (tuple): The subtuple to search for.
            wildcard (optional): A value that acts as a wildcard, matching any element in `tuple1`.
        Returns:
            bool: True if `tuple2` is a subtuple of `tuple1` considering wildcards and options, False otherwise.
        Notes:
            - If `tuple2` is longer than `tuple1`, the function returns False immediately.
            - Elements in `tuple2` can be lists, tuples, or sets of possible values.
            - If an element in `tuple2` is a list, tuple, or set containing the wildcard, it will match any corresponding element in `tuple1`.
        """

        # Check if tuple2 is longer than tuple1, if so, return False immediately
        if len(tuple2) > len(tuple1):
            return False
        
        # Iterate over tuple1 with a sliding window of size len(tuple2)
        for i in range(len(tuple1) - len(tuple2) + 1):
            match = True
            for j in range(len(tuple2)):
                elem2 = tuple2[j]
                elem1 = tuple1[i + j]
                
                # Check if it's a wildcard
                if elem2 is wildcard:
                    continue  # Skip comparison

                # Check if elem2 is a list/tuple/set of possible values
                if isinstance(elem2, (list, tuple, set)):
                    if wildcard in elem2:
                        if not (elem1 in elem2 or elem1 == elem2 or ShapeUtils.__contains_subtuple_with_wildcards_and_options(elem1, elem2)):
                            match = False
                            break
                    elif not (elem1 in elem2 or elem1 == elem2):
                        match = False
                        break

                # Check for direct comparison
                elif elem1 != elem2:
                    match = False
                    break

            if match:
                return True
        return False
    
    def extend_shape_params_with_marks(self, *shapes_to_extend: Tuple[Tuple, str]):
        """
        Extends the shape parameters with specified marks.
        This method iterates over the provided shapes and their corresponding marks,
        and for each shape, it checks if the shape is a subtuple within the existing
        parameters (considering wildcards and options). If a match is found, it updates
        the marks for that parameter.
        \nIf None is provided as a mark, the existing marks are removed.
        Args:
            *shapes_to_extend (Tuple[Tuple, str]): A variable number of tuples where each
                tuple contains a shape (as a tuple) and a mark (as a string).
        Returns:
            Self: The instance of the class.
        """

        parameters = self.test_list
        for shape, mark in shapes_to_extend:
            for param in parameters:
                if ShapeUtils.__contains_subtuple_with_wildcards_and_options(param.values, shape):
                    index = parameters.index(param)
                    current_marks = list(param.marks)
                    if mark is None:
                        current_marks = []
                    else:
                        current_marks.append(mark)
                    parameters.remove(param)
                    values = param.values
                    param_to_insert = pytest.param(*values, marks=current_marks, id=param.id)
                    parameters.insert(index, param_to_insert)
        return self

    @staticmethod
    def get_default_df_param() -> pytest.param:
        """
        Returns a pytest parameter with a default data format.
        This function creates and returns a pytest parameter with the default
        data format set to `forge.DataFormat.Float16_b`. The parameter is 
        identified with the id "dev-data-format=Float16_b".
        Returns:
            pytest.param: A pytest parameter with the default data format.
        """

        return pytest.param(forge.DataFormat.Float16_b, id="dev-data-format=Float16_b")

    @staticmethod
    def get_default_mf_param() -> pytest.param:
        """
        Returns a pytest parameter with a default MathFidelity value.
        This function creates and returns a pytest parameter with the 
        MathFidelity value set to HiFi4. The parameter is also given an 
        identifier "math-fidelity=HiFi4" for easier identification in test 
        outputs.
        Returns:
            pytest.param: A pytest parameter with MathFidelity.HiFi4 and 
                          an identifier "math-fidelity=HiFi4".
        """

        return pytest.param(forge.MathFidelity.HiFi4, id="math-fidelity=HiFi4")

    def add_df_mf_params(self, values: tuple, id: str):
        """
        Adds a list of pytest parameters to the test list with various data formats and math fidelities.
        These parameters are used to test DataFormat and MathFidelity combinations.
        Args:
            values (tuple): A tuple of values to be used as arguments for the pytest parameters.
            id (str): A string identifier to be used in the id of each pytest parameter.
        Returns:
            Self: The instance of the class.
        """

        result = []
        result.append(pytest.param(*values, forge.DataFormat.Float16_b, forge.MathFidelity.LoFi,  id=f"{id}-dev_data_format=Float16_b-math_fidelity=LoFi"))
        result.append(pytest.param(*values, forge.DataFormat.Float16_b, forge.MathFidelity.HiFi2, id=f"{id}-dev_data_format=Float16_b-math_fidelity=HiFi2"))
        result.append(pytest.param(*values, forge.DataFormat.Float16_b, forge.MathFidelity.HiFi3, id=f"{id}-dev_data_format=Float16_b-math_fidelity=HiFi3"))
        result.append(pytest.param(*values, forge.DataFormat.Float16_b, forge.MathFidelity.HiFi4, id=f"{id}-dev_data_format=Float16_b-math_fidelity=HiFi4"))

        result.append(pytest.param(*values, forge.DataFormat.Bfp2,      forge.MathFidelity.HiFi4, id=f"{id}-dev_data_format=Bfp2-math_fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Bfp2_b,    forge.MathFidelity.HiFi4, id=f"{id}-dev_data_format=Bfp2_b-math_fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Bfp4,      forge.MathFidelity.HiFi4, id=f"{id}-dev_data_format=Bfp4-math_fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Bfp4_b,    forge.MathFidelity.HiFi4, id=f"{id}-dev_data_format=Bfp4_b-math_fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Bfp8,      forge.MathFidelity.HiFi4, id=f"{id}-dev_data_format=Bfp8-math_fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Bfp8_b,    forge.MathFidelity.HiFi4, id=f"{id}-dev_data_format=Bfp8_b-math_fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Float16,   forge.MathFidelity.HiFi4, id=f"{id}-dev_data_format=Float16-math_fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Float16_b, forge.MathFidelity.HiFi4, id=f"{id}-dev_data_format=Float16_b-math_fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Float32,   forge.MathFidelity.HiFi4, id=f"{id}-dev_data_format=Float32-math_fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Int8,      forge.MathFidelity.HiFi4, id=f"{id}-dev_data_format=Int8-math_fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.Lf8,       forge.MathFidelity.HiFi4, id=f"{id}-dev_data_format=Lf8-math_fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.RawUInt16, forge.MathFidelity.HiFi4, id=f"{id}-dev_data_format=RawUInt16-math_fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.RawUInt32, forge.MathFidelity.HiFi4, id=f"{id}-dev_data_format=RawUInt32-math_fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.RawUInt8,  forge.MathFidelity.HiFi4, id=f"{id}-dev_data_format=RawUInt8-math_fidelity=HiFi4"))
        result.append(pytest.param(*values, forge.DataFormat.UInt16,    forge.MathFidelity.HiFi4, id=f"{id}-dev_data_format=UInt16-math_fidelity=HiFi4"))

        self.test_list.extend(result)

        return self

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
        self.current_value = self.rng.randint(0, self.max_limit)
        return self.current_value < self.current_limit

    def limit_info(self) -> str:
        '''Return the rate limit info for previous operation'''
        if self.current_value < self.current_limit:
            return f"{self.current_value} < {self.current_limit}"
        else:
            return f"{self.current_value} >= {self.current_limit}"
