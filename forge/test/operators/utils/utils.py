# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Operator test utilities

import random
import os
import sys
import forge
import torch
import pytest


from enum import Enum
from dataclasses import dataclass
from loguru import logger
from typing import Optional, List, Dict, Type, Union

from forge import ForgeModule, Module, DepricatedVerifyConfig
from forge.op_repo import TensorShape
from forge.verify import TestKind  # , verify_module
from forge.config import _get_global_compiler_config
from forge._C import MathFidelity

from forge.verify.config import VerifyConfig

from .compat import TestDevice
from .compat import (
    create_torch_inputs,
    verify_module_for_inputs,
    verify_module_for_inputs_deprecated,
    verify_module_for_inputs_torch,
)
from .datatypes import ValueRanges
from .features import TestFeaturesConfiguration


# All supported framework model types
FrameworkModelType = Union[
    Type[torch.nn.Module],
]


class ShapeUtils:
    @staticmethod
    def switch_last_two(t):
        if len(t) < 2:
            return t  # If tuple has less than 2 elements, return it as is
        return t[:-2] + (t[-1], t[-2])

    @staticmethod
    def reduce_microbatch_size(shape: TensorShape) -> TensorShape:
        """
        Reduce microbatch dimension of a shape to 1
        Usually used for calculating shape of a constant tensor
        """
        return (1,) + shape[1:]

    @staticmethod
    def extend_shapes_with_id(shapes: List[TensorShape]) -> List[TensorShape]:
        """
        Extend shapes with an id
        """
        shapes_with_ids = list()
        for shape in shapes:
            if type(shape) is tuple:
                shapes_with_ids.append(pytest.param(shape, marks=tuple(), id=f"shape={shape}"))
            else:
                shapes_with_ids.append(pytest.param(shape.values[0], marks=shape.marks, id=f"shape={shape.values[0]}"))
        return shapes_with_ids


@dataclass(frozen=True)
class InputSourceFlag:
    """Dataclass for specifying compiler flags for specific input source"""

    input_queues_on_host: bool
    set_default_dram_parameters: bool
    default_dram_parameters: Optional[bool]


class InputSourceFlags(Enum):
    """Enums defining input source flags"""

    FROM_HOST = InputSourceFlag(True, False, None)
    FROM_DRAM = InputSourceFlag(False, False, None)
    FROM_DRAM_PROLOGUED = InputSourceFlag(False, True, False)
    FROM_DRAM_NOT_PROLOGUED = InputSourceFlag(False, True, True)
    FROM_DRAM_PROLOGUE_MICROBATCH_SIZE = InputSourceFlag(False, True, None)


class CompilerUtils:
    """Utility functions for Forge compiler configuration"""

    @staticmethod
    def set_input_source(input_source_flag: InputSourceFlag):
        """Set compiler configuration for input source"""
        compiler_cfg = _get_global_compiler_config()
        compiler_cfg.input_queues_on_host = input_source_flag.input_queues_on_host
        if input_source_flag.set_default_dram_parameters:
            compiler_cfg.default_dram_parameters = input_source_flag.default_dram_parameters

    @staticmethod
    def set_math_fidelity(math_fidelity: MathFidelity):
        """Set compiler configuration for math fidelity"""
        compiler_cfg = _get_global_compiler_config()
        compiler_cfg.default_math_fidelity = math_fidelity


class DeviceUtils:
    """Utility functions for Forge verification"""

    @staticmethod
    def warm_reset():
        reset_command = "/home/software/syseng/wh/tt-smi -lr all wait -er"
        os.system(reset_command)


class VerifyUtils:
    """Utility functions for Forge verification"""

    @classmethod
    def verify(
        cls,
        model: Module,
        test_device: TestDevice,
        input_shapes: List[TensorShape],
        input_params: List[Dict] = [],
        pcc: Optional[float] = None,
        input_source_flag: InputSourceFlags = None,
        dev_data_format: forge.DataFormat = None,
        convert_to_forge: bool = True,  # explicit conversion to forge data format
        math_fidelity: forge.MathFidelity = None,
        value_range: Optional[ValueRanges] = None,
        random_seed: Optional[int] = None,
        warm_reset: bool = False,
        deprecated_verification: bool = True,
        verify_config: Optional[VerifyConfig] = VerifyConfig(),
        skip_forge_verification: bool = TestFeaturesConfiguration.SKIP_FORGE_VERIFICATION,
    ):
        """Perform Forge verification on the model

        Args:
            model: Forge model
            test_device: TestDevice
            input_shapes: List of input shapes
            input_params: List of input parameters
            pcc: PCC value for verification
            input_source_flag: Input source flag
            dev_data_format: Data format
            math_fidelity: Math fidelity
            value_range: Value range of input tensors
            random_seed: Random seed
            warm_reset: Warm reset the device before verification
            deprecated_verification: Use deprecated verification method
            verify_config: Verification configuration
            skip_forge_verification: Skip verification with Forge module
        """

        cls.setup(
            input_source_flag=input_source_flag,
            math_fidelity=math_fidelity,
            warm_reset=warm_reset,
        )

        inputs = cls.create_torch_inputs(
            input_shapes=input_shapes,
            dev_data_format=dev_data_format,
            value_range=value_range,
            random_seed=random_seed,
        )

        if deprecated_verification:
            cls.verify_module_for_inputs_deprecated(
                model=model,
                inputs=inputs,
                pcc=pcc,
                dev_data_format=dev_data_format,
                convert_to_forge=convert_to_forge,
            )
        elif skip_forge_verification:
            verify_module_for_inputs_torch(
                model=model,
                inputs=inputs,
                verify_config=verify_config,
            )
        else:
            cls.verify_module_for_inputs(
                model=model,
                inputs=inputs,
                verify_config=verify_config,
                dev_data_format=dev_data_format,
                convert_to_forge=convert_to_forge,
            )

    @classmethod
    def setup(
        cls,
        input_source_flag: InputSourceFlags = None,
        math_fidelity: forge.MathFidelity = None,
        warm_reset: bool = False,
    ):
        if warm_reset:
            DeviceUtils.warm_reset()

        if input_source_flag:
            CompilerUtils.set_input_source(input_source_flag.value)

        if math_fidelity:
            CompilerUtils.set_math_fidelity(math_fidelity)

        # if dev_data_format:
        #     input_params.append({"dev_data_format": dev_data_format})

    @classmethod
    def create_torch_inputs(
        cls,
        input_shapes: List[TensorShape],
        dev_data_format: forge.DataFormat = None,
        value_range: Optional[ValueRanges] = None,
        random_seed: Optional[int] = None,
    ) -> List[torch.Tensor]:

        inputs = create_torch_inputs(
            input_shapes=input_shapes,
            dev_data_format=dev_data_format,
            value_range=value_range,
            random_seed=random_seed,
        )

        return inputs

    @classmethod
    def verify_module_for_inputs_deprecated(
        cls,
        model: Module,
        inputs: List[torch.Tensor],
        pcc: Optional[float] = None,
        dev_data_format: forge.DataFormat = None,
        convert_to_forge: bool = True,  # explicit conversion to forge data format
    ):

        verify_module_for_inputs_deprecated(
            model=model,
            inputs=inputs,
            pcc=pcc,
            dev_data_format=dev_data_format,
            convert_to_forge=convert_to_forge,
        )

    @classmethod
    def verify_module_for_inputs(
        cls,
        model: Module,
        inputs: List[torch.Tensor],
        verify_config: Optional[VerifyConfig] = VerifyConfig(),
        dev_data_format: forge.DataFormat = None,
        convert_to_forge: bool = True,  # explicit conversion to forge data format
    ):

        verify_module_for_inputs(
            model=model,
            inputs=inputs,
            verify_config=verify_config,
            dev_data_format=dev_data_format,
            convert_to_forge=convert_to_forge,
        )


class LoggerUtils:
    """Utility functions for logging"""

    @staticmethod
    def set_log_level(package_name: str, level: str):
        """Set log level for package_name and its subpackages

        Args:
            package_name (str): package name
            level (str): log level
        """
        logger.add(sys.stdout, level=level, filter=lambda record: record["name"].startswith(package_name))


class RateLimiter:
    """Rate limiter class to limit the number of allowed operations by a rate limit factor"""

    def __init__(self, rng: random.Random, max_limit: int, current_limit: int):
        self.rng = rng
        self.max_limit = max_limit
        self.current_limit = current_limit
        self.current_value: int = None

    def is_allowed(self) -> bool:
        """Check if the operation is allowed by the rate limit factor and current random value"""
        self.current_value = self.rng.randint(1, self.max_limit)
        return self.current_value <= self.current_limit

    def limit_info(self) -> str:
        """Return the rate limit info for previous operation"""
        if self.current_value < self.current_limit:
            return f"{self.current_value} <= {self.current_limit}"
        else:
            return f"{self.current_value} > {self.current_limit}"
