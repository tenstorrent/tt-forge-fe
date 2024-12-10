# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Tests for testing of element-wise binary operators
#
# In this test we test pytorch binary operators

# GENERAL OP SUPPORT TEST PLAN:
# 1. Operand type - any supported type
# 2. Operand source(s):
# (+)  2.1 From another op
#       - Operator -> input
# (+)  2.2 From DRAM queue
#       - Operator is first node in network
#       - Input_queue flag = false
# (+)  2.3 Const Inputs (const eval pass)
#       - Operator where all inputs are constants.
# (+)  2.4 From host
#       - Input tensor as input of network
#       - Operator is first node in network
#       - Input_queue flag = true
# 3 Operand shapes type(s):
# (+)  3.1 Full tensor (i.e. full expected shape)
#       - 3-4 by default P1 (high prioriy)
#       - 2, 5, ++ include P2 (lower prioriy)
# (+)  3.2 Tensor reduce on one or more dims to 1
#       - Vector
#       - Only one dim is not equal to 1
# (/)  3.3 Scalar P2
#       - Create tensor of dimension equal to 0 (tensor from scalar) or just to use scalar as simple value
# 4. Operand / output size of dimensions (few examples of each, 10 values total)
# (+)  4.1 Divisible by 32
# (+)  4.2 Prime numbers
# (+)  4.3 Very large (thousands, 10s of thousands)
#       - 100x100, 100x1000
#       - maybe nightly only
# (+)  4.4 Extreme ratios between height/width
#      4.5 ...probably many more interesting combinations here
# 5. Data format - all supported formats
# (/)  5.1 Output DF
# (/)  5.2 Intermediate DF
# (/)  5.3 Accumulation DF
# (+)  5.4 Operand DFs
#       - Fix HiFi4 for math fidelity value
# (+) 6. Math fidelity - LoFi, HiFi2a, Hifi2b, Hifi3, Hifi4
#       - Fix fp16b (default) for data format value
# (/) 7. Special attributes - if applicable.. like approx_mode for Exp, for example
# (/) 8. Special cases - if applicable
# 9. Variable number of operands - if applicable
# (/) Few representative values
# (/) Reuse inputs for selected operators


from typing import Callable, List, Tuple, Dict, Union, Optional
from loguru import logger

import forge
import torch

from test.operators.utils import ValueRanges
from test.operators.utils import InputSourceFlags, VerifyUtils
from test.operators.utils import InputSource
from test.operators.utils import TestVector
from test.operators.utils import TestCollection
from test.operators.utils import TestPlan
from test.operators.utils import TestSuite
from test.operators.utils import TestResultFailing
from test.operators.utils import TestCollectionCommon
from test.operators.utils import FailingReasons
from test.operators.utils.compat import TestDevice

from forge.op_repo import TensorShape

from .models import ModelFromAnotherOp, ModelDirect, ModelConstEvalPass
from .failing_rules import FailingRulesData


class DivVerifyUtils(VerifyUtils):
    @classmethod
    def create_torch_inputs(
        cls, input_shapes, dev_data_format=None, value_range=None, random_seed=None
    ) -> List[torch.Tensor]:
        inputs = super().create_torch_inputs(input_shapes, dev_data_format, value_range, random_seed)

        # Avoid zero value in the second operand to avoid division by zero
        tensor = inputs[1]
        tensor = torch.where(tensor == 0, torch.tensor(1), tensor)
        inputs[1] = tensor

        return inputs


class TestVerification:

    MODEL_TYPES = {
        InputSource.FROM_ANOTHER_OP: ModelFromAnotherOp,
        InputSource.FROM_HOST: ModelDirect,
        InputSource.FROM_DRAM_QUEUE: ModelDirect,
        InputSource.CONST_EVAL_PASS: ModelConstEvalPass,
    }

    @classmethod
    def verify(
        cls,
        test_device: TestDevice,
        test_vector: TestVector,
        value_range: Optional[ValueRanges] = None,
        VerifyUtils=VerifyUtils,
        # number_of_operands: int = 2,
        # input_params: List[Dict] = [],
    ):
        """Common verification function for all tests"""

        number_of_operands: int = 2
        input_params: List[Dict] = []

        warm_reset = False

        input_source_flag: InputSourceFlags = None
        if test_vector.input_source in (InputSource.FROM_DRAM_QUEUE,):
            input_source_flag = InputSourceFlags.FROM_DRAM

        dev_data_format = test_vector.dev_data_format
        # if test_vector.dev_data_format is not None:
        #     dev_data_format = test_vector.dev_data_format
        # else:
        #     dev_data_format = TestCollectionCommon.single.dev_data_formats[0]

        if dev_data_format in TestCollectionCommon.int.dev_data_formats:
            value_range = ValueRanges.LARGE

        if value_range is None:
            value_range = ValueRanges.SMALL

        operator = getattr(torch, test_vector.operator)

        kwargs = test_vector.kwargs if test_vector.kwargs else {}

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        pytorch_model = model_type(
            operator=operator, opname=test_vector.operator, shape=test_vector.input_shape, kwargs=kwargs
        )
        # forge_model = forge.PyTorchModule(pytorch_model.model_name, pytorch_model)

        input_shapes = tuple([test_vector.input_shape for _ in range(number_of_operands)])
        logger.trace(f"***input_shapes: {input_shapes}")

        VerifyUtils.verify(
            model=pytorch_model,
            test_device=test_device,
            input_shapes=input_shapes,
            input_params=input_params,
            input_source_flag=input_source_flag,
            dev_data_format=dev_data_format,
            math_fidelity=test_vector.math_fidelity,
            value_range=value_range,
            pcc=test_vector.pcc,
            warm_reset=warm_reset,
        )


class TestParamsData:

    __test__ = False  # Avoid collecting TestParamsData as a pytest test

    no_kwargs = [
        None,
    ]

    kwargs_alpha_int = [
        {"alpha": 1},
        {"alpha": -37},  # TODO test this
        {"alpha": 37},
        {},
    ]

    kwargs_alpha_float = [
        {"alpha": -37},  # TODO test this
        {"alpha": 1},  # TODO test this
        {"alpha": 37},
        {"alpha": 0.17234435},
        {"alpha": 589.34546459345},
        # { "alpha": None },
        {},
    ]

    kwargs_rounding_modes = [
        {"rounding_mode": "trunc"},
        {"rounding_mode": "floor"},
        {"rounding_mode": None},
        {},
    ]

    @classmethod
    def generate_kwargs_alpha(cls, test_vector: TestVector):
        if test_vector.dev_data_format in TestCollectionCommon.int.dev_data_formats:
            return cls.kwargs_alpha_int
        else:
            return cls.kwargs_alpha_float


class TestCollectionData:

    __test__ = False  # Avoid collecting TestCollectionData as a pytest test

    implemented = TestCollection(
        operators=[
            "add",  #                   #00
            "div",  #                   #01
            # "divide",  #              #02     - Alias for div.
            "mul",  #                   #03
            # "multiply",  #            #04     - Alias for mul.
            "sub",  #                   #05
            # "subtract",  #            #06     - Alias for sub.
            # "true_divide",  #         #07     - Alias for div with rounding_mode=None.
            "ge",  #                    #08
            # "greater_equal",  #       #09    - Alias for ge.
            "ne",  #                    #16                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: not_equal      # working with model const
            # "greater",  #             #18    - Alias for gt.      E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: greater
            "gt",  #                    #19                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: greater        # working with model const
            "lt",  #                    #21                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: less           # working with model const
            # "less",  #                #22    - Alias for lt.      E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: less
            "maximum",  #               #23                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: maximum        # working with model const
            "minimum",  #               #24                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: minimum        # working with model const
            # "not_equal",  #           #25    - Alias for ne.      E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: not_equal
        ],
    )

    not_implemented = TestCollection(
        operators=[
            "atan2",  #                 #00                         - NotImplementedError: The following operators are not implemented: ['aten::atan2']
            "arctan2",  #               #01                         - NotImplementedError: The following operators are not implemented: ['aten::atan2']
            "bitwise_and",  #           #02                         - RuntimeError: "bitwise_and_cpu" not implemented for 'Float'
            "bitwise_or",  #            #03                         - RuntimeError: "bitwise_or_cpu" not implemented for 'Float'
            "bitwise_xor",  #           #04                         - RuntimeError: "bitwise_xor_cpu" not implemented for 'Float'
            "bitwise_left_shift",  #    #05                         - RuntimeError: "lshift_cpu" not implemented for 'Float'
            "bitwise_right_shift",  #   #06                         - RuntimeError: "rshift_cpu" not implemented for 'Float'
            "floor_divide",  #          #07                         - AssertionError: Encountered unsupported op types. Check error logs for more details         # working with model const
            "fmod",  #                  #08                         - AssertionError: Encountered unsupported op types. Check error logs for more details         # working with model const
            "logaddexp",  #             #09                         - NotImplementedError: The following operators are not implemented: ['aten::logaddexp']
            "logaddexp2",  #            #10                         - NotImplementedError: The following operators are not implemented: ['aten::logaddexp2']
            "nextafter",  #             #11                         - NotImplementedError: The following operators are not implemented: ['aten::nextafter']
            "remainder",  #             #12                         - AssertionError: Encountered unsupported op types. Check error logs for more details         # working with model const
            "fmax",  #                  #13                         - NotImplementedError: The following operators are not implemented: ['aten::fmax']
            "fmin",  #                  #14                         - NotImplementedError: The following operators are not implemented: ['aten::fmin']
            "eq",  #                    #15                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: equal          # working with model const
            "le",  #                    #17                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: less_equal     # working with model const
            # "less_equal",  #          #20    - Alias for le.      E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: less_equal
        ],
    )

    implemented_const_eval = TestCollection(
        operators=[
            "floor_divide",
            "fmod",
            "remainder",
            "eq",
            "ne",
            "le",
            "gt",
            "lt",
            "maximum",
            "minimum",
        ],
    )

    alpha = TestCollection(
        operators=[
            "add",  #                   #00
            "sub",  #                   #05
            # "subtract",  #            #06     - Alias for sub.
        ],
    )

    rounding_mode = TestCollection(
        operators=[
            "div",  #                   #01
            # "divide",  #              #02     - Alias for div.
            # "true_divide",  #         #07     - Alias for div with rounding_mode=None.
        ],
    )

    no_params = TestCollection(
        operators=[
            "mul",  #                   #03
            # "multiply",  #            #04     - Alias for mul.
            "ge",  #                    #08
            # "greater_equal",  #       #09     - Alias for ge.
        ],
    )

    bitwise = TestCollection(
        operators=[
            "bitwise_and",
            "bitwise_or",
            "bitwise_xor",
            "bitwise_left_shift",
            "bitwise_right_shift",
        ],
    )

    all = TestCollection(
        operators=implemented.operators,
        input_sources=TestCollectionCommon.all.input_sources,
        input_shapes=TestCollectionCommon.all.input_shapes,
        dev_data_formats=TestCollectionCommon.all.dev_data_formats,
        math_fidelities=TestCollectionCommon.all.math_fidelities,
    )

    single = TestCollection(
        input_sources=TestCollectionCommon.single.input_sources,
        input_shapes=TestCollectionCommon.single.input_shapes,
        dev_data_formats=TestCollectionCommon.single.dev_data_formats,
        math_fidelities=TestCollectionCommon.single.math_fidelities,
    )


class FailingRulesUtils:
    """Helper class for building failing rules for test plans"""

    # TODO move to shared utils

    __test__ = False  # Avoid collecting FailingRulesUtils as a pytest test

    @classmethod
    def build_rules(
        cls,
        rules: List[
            Union[
                Tuple[
                    Union[Optional[InputSource], List[InputSource]],
                    Union[Optional[TensorShape], List[TensorShape]],
                    Union[Optional[forge.DataFormat], List[forge.DataFormat]],
                    Union[Optional[forge.MathFidelity], List[forge.MathFidelity]],
                    Optional[TestResultFailing],
                ],
                TestCollection,
            ]
        ],
    ) -> List[TestCollection]:
        """Convert failing rules to TestCollection(s)"""
        test_collections = [
            cls.build_rule(
                input_source=rule[0],
                input_shape=rule[1],
                dev_data_format=rule[2],
                math_fidelity=rule[3],
                result_failing=rule[4],
            )
            if isinstance(rule, tuple)
            else rule  # if rule is already TestCollection there is no need to convert it
            for rule in rules
        ]

        return test_collections

    @classmethod
    def build_rule(
        cls,
        input_source: Optional[Union[InputSource, List[InputSource]]],
        input_shape: Optional[Union[TensorShape, List[TensorShape]]],
        dev_data_format: Optional[Union[forge.DataFormat, List[forge.DataFormat]]],
        math_fidelity: Optional[Union[forge.MathFidelity, List[forge.MathFidelity]]],
        result_failing: Optional[TestResultFailing],
    ) -> TestCollection:
        """Convert failing rule tuple to TestCollection"""

        if input_source is not None and not isinstance(input_source, list):
            input_source = [input_source]
        if input_shape is not None and not isinstance(input_shape, list):
            input_shape = [input_shape]
        if dev_data_format is not None and not isinstance(dev_data_format, list):
            dev_data_format = [dev_data_format]
        if math_fidelity is not None and not isinstance(math_fidelity, list):
            math_fidelity = [math_fidelity]

        test_collection = TestCollection(
            input_sources=input_source,
            input_shapes=input_shape,
            dev_data_formats=dev_data_format,
            math_fidelities=math_fidelity,
            failing_reason=result_failing.failing_reason if result_failing is not None else None,
            skip_reason=result_failing.skip_reason if result_failing is not None else None,
        )

        return test_collection


class BinaryTestPlanBuilder:
    """Helper class for building test plans for binary operators"""

    @classmethod
    def build_test_collections(
        cls, operator: str, generate_kwargs: Optional[Callable[[TestVector], List[Dict]]] = None, quick_mix=False
    ) -> List[TestCollection]:
        """Build test plan collections for binary operator"""

        operators = [operator]

        collections = [
            # Test plan:
            # 2. Operand source(s):
            # 3. Operand shapes type(s):
            # 4. Operand / output size of dimensions
            TestCollection(
                operators=operators,
                input_sources=TestCollectionData.all.input_sources,
                input_shapes=TestCollectionData.all.input_shapes,
                kwargs=generate_kwargs,
            ),
            # Test plan:
            # 5. Data format
            TestCollection(
                operators=operators,
                input_sources=TestCollectionData.single.input_sources,
                input_shapes=TestCollectionData.single.input_shapes,
                kwargs=generate_kwargs,
                dev_data_formats=TestCollectionData.all.dev_data_formats,
                math_fidelities=TestCollectionData.single.math_fidelities,
            ),
            # Test plan:
            # 6. Math fidelity
            TestCollection(
                operators=operators,
                input_sources=TestCollectionData.single.input_sources,
                input_shapes=TestCollectionData.single.input_shapes,
                kwargs=generate_kwargs,
                dev_data_formats=TestCollectionData.single.dev_data_formats,
                math_fidelities=TestCollectionData.all.math_fidelities,
            ),
        ]

        if quick_mix:
            collections.append(
                # Quick mix
                # Extended test plan with multiple input sources, shapes and data formats
                TestCollection(
                    operators=operators,
                    input_sources=TestCollectionData.all.input_sources,
                    input_shapes=TestCollectionCommon.quick.input_shapes,
                    kwargs=generate_kwargs,
                    dev_data_formats=TestCollectionCommon.quick.dev_data_formats,
                    math_fidelities=TestCollectionData.single.math_fidelities,
                )
            )

        return collections

    @classmethod
    def build_test_plan(
        cls,
        operator: str,
        value_range: ValueRanges,
        generate_kwargs: Optional[Callable[[TestVector], List[Dict]]] = None,
        quick_mix: bool = False,
        VerifyUtils=VerifyUtils,
    ) -> List[TestCollection]:
        """Build test plan for a binary operator"""

        if generate_kwargs is None:
            generate_kwargs = lambda test_vector: TestParamsData.no_kwargs

        failing_rules = getattr(FailingRulesData, operator)

        failing_rules = FailingRulesUtils.build_rules(failing_rules)

        test_plan = TestPlan(
            verify=lambda test_device, test_vector: TestVerification.verify(
                test_device,
                test_vector,
                value_range=value_range,
                VerifyUtils=VerifyUtils,
            ),
            collections=cls.build_test_collections(operator, generate_kwargs, quick_mix),
            failing_rules=failing_rules,
        )

        return test_plan


class TestPlansData:

    __test__ = False  # Avoid collecting TestPlansData as a pytest test

    add: TestPlan = BinaryTestPlanBuilder.build_test_plan(
        "add",
        value_range=ValueRanges.LARGE,
        generate_kwargs=lambda test_vector: TestParamsData.generate_kwargs_alpha(test_vector),
        quick_mix=False,
    )

    sub: TestPlan = BinaryTestPlanBuilder.build_test_plan(
        "sub",
        value_range=ValueRanges.SMALL,
        generate_kwargs=lambda test_vector: TestParamsData.generate_kwargs_alpha(test_vector),
        quick_mix=False,
    )

    mul: TestPlan = BinaryTestPlanBuilder.build_test_plan("mul", value_range=ValueRanges.SMALL, quick_mix=False)

    div: TestPlan = BinaryTestPlanBuilder.build_test_plan(
        "div",
        value_range=ValueRanges.LARGE,
        generate_kwargs=lambda test_vector: TestParamsData.kwargs_rounding_modes,
        quick_mix=False,
        VerifyUtils=DivVerifyUtils,
    )

    ge: TestPlan = BinaryTestPlanBuilder.build_test_plan("ge", value_range=ValueRanges.SMALL, quick_mix=False)

    ne: TestPlan = BinaryTestPlanBuilder.build_test_plan("ne", value_range=ValueRanges.SMALL, quick_mix=True)

    gt: TestPlan = BinaryTestPlanBuilder.build_test_plan("gt", value_range=ValueRanges.SMALL, quick_mix=True)

    lt: TestPlan = BinaryTestPlanBuilder.build_test_plan("lt", value_range=ValueRanges.SMALL, quick_mix=True)

    maximum: TestPlan = BinaryTestPlanBuilder.build_test_plan("maximum", value_range=ValueRanges.SMALL, quick_mix=True)

    minimum: TestPlan = BinaryTestPlanBuilder.build_test_plan("minimum", value_range=ValueRanges.SMALL, quick_mix=True)

    not_implemented: TestPlan = TestPlan(
        verify=lambda test_device, test_vector: TestVerification.verify(
            test_device, test_vector, value_range=ValueRanges.SMALL
        ),
        collections=[
            # Unimplemented operators
            TestCollection(
                operators=TestCollectionData.not_implemented.operators,
                input_sources=TestCollectionData.all.input_sources,
                input_shapes=TestCollectionData.single.input_shapes,
            ),
        ],
        failing_rules=[
            # Not implemented operators
            TestCollection(
                operators=TestCollectionData.not_implemented.operators,
                failing_reason=FailingReasons.NOT_IMPLEMENTED,
            ),
            # Not implemented operators for CONST_EVAL_PASS
            # 10 operators are implemented for CONST_EVAL_PASS the are not for other input sources
            TestCollection(
                operators=TestCollectionData.implemented_const_eval.operators,
                input_sources=[
                    InputSource.CONST_EVAL_PASS,
                ],
                failing_reason=None,
            ),
        ],
    )


def get_test_plans() -> List[Union[TestPlan, TestSuite]]:
    return [
        TestPlansData.ne,
        TestPlansData.gt,
        TestPlansData.lt,
        TestPlansData.maximum,
        TestPlansData.minimum,
        TestPlansData.add,
        TestPlansData.sub,
        TestPlansData.mul,
        TestPlansData.div,
        TestPlansData.ge,
        TestPlansData.not_implemented,
    ]
