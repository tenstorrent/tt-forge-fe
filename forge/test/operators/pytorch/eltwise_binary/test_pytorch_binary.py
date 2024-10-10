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


# Run single test
# 
# To run single test use following parameters
# id         --test_id       --> None

# Examples
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-ge-FROM_HOST-None-(1, 2, 3, 4)-Float16_b-HiFi4'

# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_plan
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_skipped
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_failed
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_not_implemented
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_data_mismatch
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_unsupported
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_unsupported_fatal
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_inconsistency_order1
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_inconsistency_order2

# Collect fatal test ids
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_unsupported_fatal --collect-only

# Run fatal tests
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-add-FROM_HOST-None-(1, 2, 3, 4)-Bfp4-HiFi4'
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-add-FROM_HOST-None-(1, 2, 3, 4)-Bfp8-HiFi4'
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-add-FROM_HOST-None-(1, 2, 3, 4)-Float16-HiFi4'
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-add-FROM_HOST-None-(1, 2, 3, 4)-Lf8-HiFi4'
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-div-FROM_HOST-None-(1, 2, 3, 4)-Bfp4-HiFi4'
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-div-FROM_HOST-None-(1, 2, 3, 4)-Bfp8-HiFi4'
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-div-FROM_HOST-None-(1, 2, 3, 4)-Float16-HiFi4'
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-div-FROM_HOST-None-(1, 2, 3, 4)-Lf8-HiFi4'
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-mul-FROM_HOST-None-(1, 2, 3, 4)-Bfp4-HiFi4'
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-mul-FROM_HOST-None-(1, 2, 3, 4)-Bfp8-HiFi4'
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-mul-FROM_HOST-None-(1, 2, 3, 4)-Float16-HiFi4'
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-mul-FROM_HOST-None-(1, 2, 3, 4)-Lf8-HiFi4'
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-sub-FROM_HOST-None-(1, 2, 3, 4)-Bfp4-HiFi4'
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-sub-FROM_HOST-None-(1, 2, 3, 4)-Bfp8-HiFi4'
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-sub-FROM_HOST-None-(1, 2, 3, 4)-Float16-HiFi4'
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-sub-FROM_HOST-None-(1, 2, 3, 4)-Lf8-HiFi4'
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-ge-FROM_HOST-None-(1, 2, 3, 4)-Bfp4-HiFi4'
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-ge-FROM_HOST-None-(1, 2, 3, 4)-Bfp8-HiFi4'
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-ge-FROM_HOST-None-(1, 2, 3, 4)-Float16-HiFi4'
# pytest -svv forge/test/operators/pytorch/eltwise_binary/test_pytorch_binary.py::test_single  --test_id='no_device-ge-FROM_HOST-None-(1, 2, 3, 4)-Lf8-HiFi4'


import pytest

from typing import List, Dict, Type, Optional, Any, Generator
from loguru import logger

import random
import torch
import forge
import forge.op

from forge.op_repo import TensorShape

from test.operators.utils import InputSourceFlags, VerifyUtils
from test.operators.utils import ShapeUtils
from test.operators.utils import InputSource
from test.operators.utils import TestVector
from test.operators.utils import TestCollection
from test.operators.utils import TestPlanUtils
from test.operators.utils import TestParamsFilter
from test.operators.utils import TestResultFailing
from test.operators.utils import TestPlan
from test.operators.utils import TestCollectionCommon
from test.operators.utils import FailingReasons
from test.operators.utils.compat import TestDevice
from test.operators.utils import RateLimiter


class ModelFromAnotherOp(torch.nn.Module):

    model_name = "model_op_src_from_another_op"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelFromAnotherOp, self).__init__()
        self.testname = "Element_wise_pytorch_binary_operator_" + opname + "_test_op_src_from_another_op"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # we use Add and Subtract operators to create two operands which are inputs for the binary operator
        xx = torch.add(x, y)
        yy = torch.add(x, y) # TODO temporary we use add operator, return to sub later
        output = self.operator(xx, yy, **self.kwargs)
        return output


class ModelDirect(torch.nn.Module):

    model_name = "model_op_src_from_host"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelDirect, self).__init__()
        self.testname = "Element_wise_pytorch_binary_operator_" + opname + "_test_op_src_from_host"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        output = self.operator(x, y, **self.kwargs)
        return output


class ModelConstEvalPass(torch.nn.Module):

    model_name = "model_op_src_const_eval_pass"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelConstEvalPass, self).__init__()
        self.testname = "Element_wise_pytorch_binary_operator_" + opname + "_test_op_src_const_eval_pass"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs
        
        self.constant_shape = ShapeUtils.reduce_microbatch_size(shape)

        self.c1 = (torch.rand(*self.constant_shape) - 0.5)
        self.c2 = (torch.rand(*self.constant_shape) - 0.5)

    def forward(self, x, y):
        v1 = self.operator(self.c1, self.c2, **self.kwargs)
        # v2 and v3 consume inputs
        v2 = torch.add(x, y)
        v3 = torch.add(v1, v2)
        return v3


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
        number_of_operands: int = 2,
        input_params: List[Dict] = [],
        dry_run: bool = False,
    ):
        '''Common verification function for all tests'''

        if dry_run:
            return

        input_source_flag: InputSourceFlags = None
        if test_vector.input_source in (InputSource.FROM_DRAM_QUEUE,):
            input_source_flag = InputSourceFlags.FROM_DRAM

        operator = getattr(torch, test_vector.operator)

        kwargs = test_vector.kwargs if test_vector.kwargs else {}

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        pytorch_model = model_type(operator=operator, opname=test_vector.operator, shape=test_vector.input_shape, kwargs=kwargs)
        # forge_model = forge.PyTorchModule(pytorch_model.model_name, pytorch_model)

        input_shapes = tuple([test_vector.input_shape for _ in range(number_of_operands)])
        logger.trace(f"***input_shapes: {input_shapes}")

        VerifyUtils.verify(
            model=pytorch_model,
            test_device=test_device,
            input_shapes=input_shapes,
            input_params=input_params,
            input_source_flag=input_source_flag,
            dev_data_format=test_vector.dev_data_format,
            math_fidelity=test_vector.math_fidelity,
            pcc=test_vector.pcc,
        )


class TestParamsData:

    __test__ = False  # Avoid collecting TestParamsData as a pytest test

    test_plan: TestPlan = None

    @classmethod
    def get_params_test_plan(cls, filter: Optional[TestParamsFilter] = None):
        return TestPlanUtils.generate_params(cls.test_plan, filter)

    @classmethod
    def get_params_from_id_file(cls, test_ids_file: str, filter: Optional[TestParamsFilter] = None):
        test_plan_ids = TestPlanUtils.build_test_plan_from_id_file(test_ids_file, cls.test_plan)
        return TestPlanUtils.generate_params(test_plan_ids, filter)

    @classmethod
    def get_params_from_id_list(cls, test_ids: List[str], filter: Optional[TestParamsFilter] = None):
        test_plan_ids = TestPlanUtils.build_test_plan_from_id_list(test_ids, cls.test_plan)
        return TestPlanUtils.generate_params(test_plan_ids, filter)

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):
        if test_vector.operator in TestCollectionData.alpha.operators:
            if test_vector.dev_data_format in TestCollectionData.int.dev_data_formats:
                return [
                    { "alpha": 1 },
                    { "alpha": 37 },
                    {},
                ]
            else:
                return [
                    { "alpha": 37 },
                    { "alpha": 0.17234435 },
                    { "alpha": 589.34546459345 },
                    # { "alpha": None },
                    {},
                ]
        elif test_vector.operator in TestCollectionData.rounding_mode.operators: 
            if not (test_vector.operator == 'div' and test_vector.dev_data_format == forge.DataFormat.Bfp2):
                return [
                    { "rounding_mode": 'trunc' },
                    { "rounding_mode": 'floor' },
                    { "rounding_mode": None },
                    {},
                ]
            else:
                return [
                    {},
                ]
        else:
            return [
                None,
            ]


class TestCollectionData:

    __test__ = False  # Avoid collecting TestCollectionData as a pytest test

    implemented = TestCollection(
        operators=[
            "add",                      #00                                                   
            "div",                      #01                                                   
            # "divide",                 #02     - Alias for div.                              
            "mul",                      #03                                                   
            # "multiply",               #04     - Alias for mul.    
            "sub",                      #05    
            # "subtract",               #06     - Alias for sub.                                                   
            # "true_divide",            #07     - Alias for div with rounding_mode=None.      
            "ge",                       #08                                                  
            # "greater_equal",          #09    - Alias for ge.  
        ],
    )

    not_implemented = TestCollection(
        operators=[
            "atan2",                    #00                         - NotImplementedError: The following operators are not implemented: ['aten::atan2']
            "arctan2",                  #01                         - NotImplementedError: The following operators are not implemented: ['aten::atan2']
            "bitwise_and",              #02                         - RuntimeError: "bitwise_and_cpu" not implemented for 'Float'
            "bitwise_or",               #03                         - RuntimeError: "bitwise_or_cpu" not implemented for 'Float'
            "bitwise_xor",              #04                         - RuntimeError: "bitwise_xor_cpu" not implemented for 'Float'
            "bitwise_left_shift",       #05                         - RuntimeError: "lshift_cpu" not implemented for 'Float'
            "bitwise_right_shift",      #06                         - RuntimeError: "rshift_cpu" not implemented for 'Float'
            "floor_divide",             #07                         - AssertionError: Encountered unsupported op types. Check error logs for more details         # working with model const 
            "fmod",                     #08                         - AssertionError: Encountered unsupported op types. Check error logs for more details         # working with model const 
            "logaddexp",                #09                         - NotImplementedError: The following operators are not implemented: ['aten::logaddexp']
            "logaddexp2",               #10                         - NotImplementedError: The following operators are not implemented: ['aten::logaddexp2']
            "nextafter",                #11                         - NotImplementedError: The following operators are not implemented: ['aten::nextafter']
            "remainder",                #12                         - AssertionError: Encountered unsupported op types. Check error logs for more details         # working with model const 
            "fmax",                     #13                         - NotImplementedError: The following operators are not implemented: ['aten::fmax']
            "fmin",                     #14                         - NotImplementedError: The following operators are not implemented: ['aten::fmin']
            
            "eq",                       #15                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: equal          # working with model const
            "ne",                       #16                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: not_equal      # working with model const
            "le",                       #17                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: less_equal     # working with model const
            # "greater",                #18    - Alias for gt.      E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: greater
            "gt",                       #19                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: greater        # working with model const
            # "less_equal",             #20    - Alias for le.      E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: less_equal
            "lt",                       #21                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: less           # working with model const
            # "less",                   #22    - Alias for lt.      E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: less
            "maximum",                  #23                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: maximum        # working with model const
            "minimum",                  #24                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: minimum        # working with model const
            # "not_equal",              #25    - Alias for ne.      E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: not_equal
        ],
    )

    implemented_const_eval = TestCollection(
        operators=[
            'floor_divide',
            'fmod',
            'remainder',
            'eq',
            'ne',
            'le',
            'gt',
            'lt',
            'maximum',
            'minimum',
        ],
    )

    alpha = TestCollection(
        operators=[
            "add",                      #00                                                   
            "sub",                      #05    
        ],
    )

    rounding_mode = TestCollection(
        operators=[
            "div",                      #01                                                   
            # "divide",                 #02     - Alias for div.                              
        ],
    )

    no_params = TestCollection(
        operators=[
            # "add",                      #00                                                   
            # "div",                      #01                                                   
            # # "divide",                 #02     - Alias for div.                              
            "mul",                      #03                                                   
            # "multiply",               #04     - Alias for mul.    
            # "sub",                      #05    
            # # "subtract",               #06     - Alias for sub.                                                   
            # "true_divide",            #07     - Alias for div with rounding_mode=None.      
            "ge",                       #08                                                  
            # "greater_equal",          #09    - Alias for ge.  
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

    bool = TestCollection(
        dev_data_formats=[
            forge.DataFormat.Bfp2,  # TODO div process as bool but add as float
        ],
    )

    int = TestCollection(
        dev_data_formats=[
            forge.DataFormat.Int8,
            forge.DataFormat.RawUInt16,
            forge.DataFormat.RawUInt32,
            forge.DataFormat.RawUInt8,
            forge.DataFormat.UInt16,
        ],
    )


TestParamsData.test_plan = TestPlan(
    collections = [
        # Test plan: 
        # 2. Operand source(s):
        # 3. Operand shapes type(s):
        # 4. Operand / output size of dimensions
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test plan: 
        # 5. Data format
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=TestCollectionData.all.dev_data_formats,
            math_fidelities=TestCollectionData.single.math_fidelities,
        ),
        # Test plan: 
        # 6. Math fidelity
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=TestCollectionData.single.dev_data_formats,
            math_fidelities=TestCollectionData.all.math_fidelities
        ),
        # Unimplemented operators
        TestCollection(
            operators=TestCollectionData.not_implemented.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
        ),
    ],
    failing_rules = [
        # Unsupported datatypes
        TestCollection(
            dev_data_formats=[
                forge.DataFormat.Bfp2,
                forge.DataFormat.Bfp4,
                forge.DataFormat.Bfp8,
                forge.DataFormat.Float16,
                forge.DataFormat.Int8,
                forge.DataFormat.Lf8,
                forge.DataFormat.RawUInt16,
                forge.DataFormat.RawUInt32,
                forge.DataFormat.RawUInt8,
                forge.DataFormat.UInt16,
            ],

            failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        ),
        # Segmentation faults or fatal python errors for Unsupported datatypes
        TestCollection(
            operators=TestCollectionData.implemented.operators,
            # operators=[
            #     "add",
            #     "sub",
            #     "mul",
            #     "div",
            #     "ge",
            # ],
            input_sources=None,
            input_shapes=None,
            dev_data_formats=[
                # forge.DataFormat.Bfp2,
                forge.DataFormat.Bfp4,
                forge.DataFormat.Bfp8,
                forge.DataFormat.Float16,
                # forge.DataFormat.Int8,
                forge.DataFormat.Lf8,
                # forge.DataFormat.RawUInt16,
                # forge.DataFormat.RawUInt32,
                # forge.DataFormat.RawUInt8,
                # forge.DataFormat.UInt16,
            ],

            failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
            skip_reason=FailingReasons.FATAL_ERROR,
        ),
        # PCC check fails for buggy shapes for all models
        TestCollection(
            input_shapes=[
                (1, 1),
            ],

            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # Exception from DATA_MISMATCH
        TestCollection(
            operators=['sub'],
            input_sources=[InputSource.FROM_ANOTHER_OP, ],
            input_shapes=[(1, 1), ],

            failing_reason=None
        ),
        # Exception from DATA_MISMATCH
        TestCollection(
            operators=['ge'],
            input_sources=[InputSource.FROM_ANOTHER_OP, InputSource.FROM_HOST, InputSource.FROM_DRAM_QUEUE, ],
            input_shapes=[(1, 1), ],

            failing_reason=None
        ),
        # PCC check fails for buggy shapes for model ModelConstEvalPass
        TestCollection(
            operators=None,
            input_sources=[InputSource.CONST_EVAL_PASS, ],
            input_shapes=[
                (11, 45, 17),
                (10, 1000, 100),
                (10, 10000, 1),
                (32, 32, 64),
                # fail only for const eval pass not for other models
                (2, 3, 4),
                (11, 1, 23),
                (11, 64, 1),
                (100, 100, 100),
                (64, 160, 96),
                (11, 17, 41),
                (13, 89, 3),
            ],

            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # Exception from DATA_MISMATCH
        TestCollection(
            operators=['mul'],
            input_sources=[InputSource.CONST_EVAL_PASS, ],
            input_shapes=[(2, 3, 4), ],

            failing_reason=None
        ),
        # PCC check fails for buggy shapes for div
        TestCollection(
            operators=["div", "divide", "true_divide", ],
            input_sources=[InputSource.FROM_HOST, InputSource.FROM_DRAM_QUEUE, ],
            input_shapes=[
                # (1, 4),  # TODO remove fixed
                # (1, 3),  # TODO remove fixed
                # (3, 4),  # TODO remove fixed
                # (1, 3, 4),  # TODO remove fixed
                (12, 64, 160, 96),
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # PCC check fails for buggy shapes for div
        TestCollection(
            operators=["div", "divide", "true_divide", ],
            input_sources=[InputSource.CONST_EVAL_PASS, ],
            input_shapes=[
                # (1, 17),  # TODO remove fixed
                # (45, 17),  # TODO remove fixed
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # PCC check fails for buggy shapes for ge
        TestCollection(
            operators=["ge", "greater_equal", ],
            input_sources=[InputSource.FROM_HOST, InputSource.FROM_DRAM_QUEUE, ],
            input_shapes=[
                (1, 1000),
                (5, 11, 64, 1),

                # fail when dtype=float32 or generator
                # (17, 41),
                # (89, 3),
                # (1, 17, 41),
                # (1, 89, 3),
            ],

            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # PCC check fails for buggy shapes for div
        TestCollection(
            operators=["div", ],
            input_sources=[InputSource.CONST_EVAL_PASS, ],
            kwargs=[
                {
                    "rounding_mode": 'trunc',
                }
            ],
            # TODO remove fixed
            dev_data_formats=[
                # forge.DataFormat.Bfp2,  # PCC check failed for some shapes
                # forge.DataFormat.Bfp2_b,
                # forge.DataFormat.Bfp8,
                # forge.DataFormat.Bfp8_b,  # PCC check failed for some shapes
                # forge.DataFormat.Float32,
                # forge.DataFormat.Int8,
                # forge.DataFormat.Lf8,
                # forge.DataFormat.RawUInt16,
            ],

            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # Not implemented operators
        TestCollection(
            operators=TestCollectionData.not_implemented.operators,

            failing_reason=FailingReasons.NOT_IMPLEMENTED,
        ),
        # Not implemented operators for CONST_EVAL_PASS
        # 10 operators are implemented for CONST_EVAL_PASS the are not for other input sources
        TestCollection(
            operators=TestCollectionData.implemented_const_eval.operators,
            input_sources=[InputSource.CONST_EVAL_PASS, ],

            failing_reason=None,
        ),
        TestCollection(
            operators=['sub'],
            dev_data_formats=[forge.DataFormat.Bfp2, ],

            failing_reason=FailingReasons.NOT_IMPLEMENTED,
            skip_reason=FailingReasons.NOT_IMPLEMENTED,
        ),
    ]
)



DRY_RUN = False
# DRY_RUN = True


@pytest.mark.parametrize("test_vector", TestParamsData.get_params_test_plan())
def test_plan(test_vector: TestVector, test_device):
    TestVerification.verify(test_device=test_device, test_vector=test_vector, dry_run=DRY_RUN)
# 1480 passed, 20 xfailed, 2 warnings in 529.46s (0:08:49) 
# 4 failed, 1352 passed, 212 xfailed, 115 xpassed, 2 warnings in 590.56s (0:09:50)
# 1 failed, 4041 passed, 20 skipped, 321 xfailed, 2 warnings in 1510.10s (0:25:10)


@pytest.mark.parametrize("test_vector", TestParamsData.get_params_test_plan(filter=TestParamsFilter(
        allow=lambda test_vector: test_vector.failing_result is not None and test_vector.failing_result.failing_reason is not None and test_vector.failing_result.skip_reason is None,
)))
def test_failed(test_vector: TestVector, test_device):
    TestVerification.verify(test_device=test_device, test_vector=test_vector, dry_run=DRY_RUN)


@pytest.mark.parametrize("test_vector", TestParamsData.get_params_test_plan(filter=TestParamsFilter(
        allow=lambda test_vector: test_vector.failing_result is not None and test_vector.failing_result.skip_reason is not None,
)))
def test_skipped(test_vector: TestVector, test_device):
    TestVerification.verify(test_device=test_device, test_vector=test_vector, dry_run=DRY_RUN)


@pytest.mark.parametrize("test_vector", TestParamsData.get_params_test_plan(filter=TestParamsFilter(
        allow=lambda test_vector: test_vector.failing_result is not None and test_vector.failing_result.failing_reason == FailingReasons.UNSUPPORTED_DATA_FORMAT and test_vector.failing_result.skip_reason == FailingReasons.FATAL_ERROR,
        # indices=1,
)))
def test_unsupported_fatal(test_vector: TestVector, test_device):
    TestVerification.verify(test_device=test_device, test_vector=test_vector, dry_run=DRY_RUN)


@pytest.mark.parametrize("test_vector", TestParamsData.get_params_test_plan(filter=TestParamsFilter(
        allow=lambda test_vector: test_vector.failing_result is not None and test_vector.failing_result.failing_reason == FailingReasons.UNSUPPORTED_DATA_FORMAT and test_vector.failing_result.skip_reason is None,
)))
def test_unsupported(test_vector: TestVector, test_device):
    TestVerification.verify(test_device=test_device, test_vector=test_vector, dry_run=DRY_RUN)


@pytest.mark.parametrize("test_vector", TestParamsData.get_params_test_plan(filter=TestParamsFilter(
        allow=lambda test_vector: test_vector.failing_result is not None and test_vector.failing_result.failing_reason == FailingReasons.NOT_IMPLEMENTED and test_vector.failing_result.skip_reason is None,
)))
def test_not_implemented(test_vector: TestVector, test_device):
    TestVerification.verify(test_device=test_device, test_vector=test_vector, dry_run=DRY_RUN)


@pytest.mark.parametrize("test_vector", TestParamsData.get_params_test_plan(filter=TestParamsFilter(
        allow=lambda test_vector: test_vector.failing_result is not None and test_vector.failing_result.failing_reason == FailingReasons.DATA_MISMATCH and test_vector.failing_result.skip_reason is None,
)))
def test_data_mismatch(test_vector: TestVector, test_device):
    TestVerification.verify(test_device=test_device, test_vector=test_vector, dry_run=DRY_RUN)


test_ids_inconsistency = [
    "no_device-div-CONST_EVAL_PASS-{}-(13, 89, 3)-None-None",
    "no_device-mul-CONST_EVAL_PASS-{}-(2, 3, 4)-None-None",
    "no_device-mul-CONST_EVAL_PASS-{}-(11, 45, 17)-None-None",
]


@pytest.mark.parametrize("test_vector", TestParamsData.get_params_from_id_list(test_ids_inconsistency, filter=TestParamsFilter(
        # reversed=True,
        log=True,
)))
def test_inconsistency_order1(test_vector: TestVector, test_device):
    TestVerification.verify(test_device=test_device,test_vector=test_vector, dry_run=DRY_RUN)


@pytest.mark.parametrize("test_vector", TestParamsData.get_params_from_id_list(test_ids_inconsistency, filter=TestParamsFilter(
        reversed=True,
        log=True,
)))
def test_inconsistency_order2(test_vector: TestVector, test_device):
    TestVerification.verify(test_device=test_device,test_vector=test_vector, dry_run=DRY_RUN)


def test_single(single_test_vector: TestVector, test_device):
    TestVerification.verify(test_device=test_device,test_vector=single_test_vector, dry_run=DRY_RUN)

