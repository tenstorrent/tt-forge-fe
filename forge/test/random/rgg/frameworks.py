# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# In depth testing of Forge models with one randomly selected operation


from enum import Enum

from loguru import logger
from typing import Tuple, Type
from copy import copy

from .datatypes import Framework, ModelBuilder
from .shapes import OperatorShapes

from .forge.model import ForgeModelBuilder
from .pytorch.model import PyTorchModelBuilder

from forge.op_repo import forge_operator_repository
from forge.op_repo import pytorch_operator_repository
from forge.op_repo import OperatorDefinition
from forge.op_repo import OperatorRepository


class FrameworkTestUtils:
    @classmethod
    def copy_framework(cls, framework: Framework, framework_name: str = None) -> Framework:
        framework = copy(framework)
        if framework_name is not None:
            framework.framework_name = framework_name
        framework.operator_repository = copy(framework.operator_repository)
        return framework

    @classmethod
    def skip_operators(cls, framework: Framework, skip_operators: Tuple[str] = []) -> None:
        allow_operators = (framework.operator_list - skip_operators).operators
        framework.operator_repository.operators = [
            op for op in framework.operator_repository.operators if op.name in allow_operators
        ]

    @classmethod
    def allow_operators(cls, framework: Framework, allow_operators: Tuple[str] = []) -> None:
        allow_operators = (framework.operator_list * allow_operators).operators
        framework.operator_repository.operators = [
            op for op in framework.operator_repository.operators if op.name in allow_operators
        ]

    @classmethod
    def copy_operator(cls, framework: Framework, operator_name: str) -> OperatorDefinition:
        operators = framework.operator_repository.operators

        i, operator = next(
            ((i, operator) for i, operator in enumerate(operators) if operator.name == operator_name), (None, None)
        )
        if not operator:
            return None

        operator = copy(operator)
        operators[i] = operator
        return operator

    @classmethod
    def set_calc_input_shapes(cls, framework: Framework, allow_operators: Tuple[str] = []) -> None:
        """Implicitly set calc_input_shapes for all operators in the framework"""
        logger.debug(f"Setting calc_input_shapes for framework {framework.framework_name}")
        for operator in framework.operator_repository.operators:
            function_name = f"{operator.name}_inputs"
            if function_name in OperatorShapes.__dict__:
                logger.debug(f"Found method {function_name} for {operator.name}")
                operator.calc_input_shapes = OperatorShapes.__dict__[function_name]
            else:
                operator.calc_input_shapes = OperatorShapes.same_input_shapes


class Frameworks(Enum):
    """Register of all frameworks"""

    @staticmethod
    def build_framework(
        template_name: str,
        framework_name: str,
        ModelBuilderType: Type[ModelBuilder],
        operator_repository: OperatorRepository,
    ):
        framework = Framework(
            template_name=template_name,
            framework_name=framework_name,
            ModelBuilderType=ModelBuilderType,
            operator_repository=operator_repository,
        )

        framework = FrameworkTestUtils.copy_framework(framework=framework)

        FrameworkTestUtils.set_calc_input_shapes(framework)

        return framework

    FORGE = build_framework(
        template_name="Forge",
        framework_name="Forge",
        ModelBuilderType=ForgeModelBuilder,
        operator_repository=forge_operator_repository,
    )
    PYTORCH = build_framework(
        template_name="PyTorch",
        framework_name="PyTorch",
        ModelBuilderType=PyTorchModelBuilder,
        operator_repository=pytorch_operator_repository,
    )
