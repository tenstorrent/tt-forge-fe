# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
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



import pytest

from typing import List, Dict, Type, Optional, Any
from loguru import logger

import random
import torch
import forge
import forge.op

from forge.op_repo import TensorShape

from test.operators.utils import InputSourceFlags, VerifyUtils
from test.operators.utils import ShapeUtils
from test.operators.utils import InputSource
from test.operators.utils import TestVectors
from test.operators.utils import TestResultFailing
from test.operators.utils import TestPlan
from test.operators.utils import TestParameterGenerator
from test.operators.utils import TestData
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


MODEL_TYPES = {
    InputSource.FROM_ANOTHER_OP: ModelFromAnotherOp,
    InputSource.FROM_HOST: ModelDirect,
    InputSource.FROM_DRAM_QUEUE: ModelDirect,
    InputSource.CONST_EVAL_PASS: ModelConstEvalPass,
}



def verify(
    test_device: TestDevice,
    input_source: InputSource,
    input_operator: str,
    input_shape: TensorShape,
    number_of_operands: int = 2,
    kwargs: Dict = {},
    input_params: List[Dict] = [],
    dev_data_format: forge.DataFormat = None,
    math_fidelity: forge.MathFidelity = None,
    pcc: Optional[float] = None,
):
    '''Common verification function for all tests'''

    input_source_flag: InputSourceFlags = None
    if input_source in (InputSource.FROM_DRAM_QUEUE,):
        input_source_flag = InputSourceFlags.FROM_DRAM

    operator = getattr(torch, input_operator)

    model_type = MODEL_TYPES[input_source]
    pytorch_model = model_type(operator=operator, opname=input_operator, shape=input_shape, kwargs=kwargs)
    # forge_model = forge.PyTorchModule(pytorch_model.model_name, pytorch_model)

    input_shapes = tuple([input_shape for _ in range(number_of_operands)])
    logger.trace(f"***input_shapes: {input_shapes}")

    VerifyUtils.verify(
        model=pytorch_model,
        test_device=test_device,
        input_shapes=input_shapes,
        input_params=input_params,
        input_source_flag=input_source_flag,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
        pcc=pcc,
    )


def get_eltwise_binary_ops():
    return [
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
    ]

def get_not_implemented_pytorch_binary_ops():
    return [
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
    ]


def get_input_shapes():
    return [
            # 2-dimensional shape, microbatch_size = 1:
            pytest.param((1, 4),              marks=pytest.mark.run_in_pp),  #00      # 3.1 Full tensor (i.e. full expected shape) 
            pytest.param((1, 17),             marks=pytest.mark.slow),       #01      # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((1, 23),             marks=pytest.mark.slow),       #02      # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((1, 1),              marks=pytest.mark.slow),       #03      # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((1, 100),            marks=pytest.mark.slow),       #04      # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((1, 500),            marks=pytest.mark.slow),       #05      # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((1, 1000),           marks=pytest.mark.slow),       #06      # 4.4 Extreme ratios between height/width
            pytest.param((1, 1920),           marks=pytest.mark.slow),       #07      # 4.4 Extreme ratios between height/width
            pytest.param((1, 10000),          marks=pytest.mark.slow),       #08      # 4.4 Extreme ratios between height/width
            pytest.param((1, 64),             marks=pytest.mark.run_in_pp),  #09      # 4.1 Divisible by 32
            pytest.param((1, 96),             marks=pytest.mark.slow),       #10      # 4.1 Divisible by 32
            pytest.param((1, 41),             marks=pytest.mark.slow),       #11      # 4.2 Prime numbers
            pytest.param((1, 3),              marks=pytest.mark.slow),       #12      # 4.2 Prime numbers

            # 2-dimensional shape, microbatch_size > 1:
            pytest.param((3, 4),              marks=pytest.mark.run_in_pp),  #13      # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((45, 17),            marks=pytest.mark.slow),       #14      # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((64, 1),             marks=pytest.mark.slow),       #15      # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((100, 100),          marks=pytest.mark.slow),       #16      # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((1000, 100),         marks=pytest.mark.slow),       #17      # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((10, 1000),          marks=pytest.mark.slow),       #18      # 4.4 Extreme ratios between height/width
            pytest.param((9920, 1),           marks=pytest.mark.slow),       #19      # 4.4 Extreme ratios between height/width  
            pytest.param((10000, 1),          marks=pytest.mark.slow),       #20      # 4.4 Extreme ratios between height/width 
            pytest.param((32, 64),            marks=pytest.mark.slow),       #21      # 4.1 Divisible by 32
            pytest.param((160, 96),           marks=pytest.mark.slow),       #22      # 4.1 Divisible by 32
            pytest.param((17, 41),            marks=pytest.mark.run_in_pp),  #23      # 4.2 Prime numbers
            pytest.param((89, 3),             marks=pytest.mark.slow),       #24      # 4.2 Prime numbers

            # 3-dimensional shape, microbatch_size = 1:
            pytest.param((1, 3, 4),           marks=pytest.mark.run_in_pp),  #25     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((1, 45, 17),         marks=pytest.mark.slow),       #26     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((1, 1, 23),          marks=pytest.mark.slow),       #27     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((1, 64, 1),          marks=pytest.mark.slow),       #28     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((1, 100, 100),       marks=pytest.mark.slow),       #29     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((1, 1000, 100),      marks=pytest.mark.slow),       #30     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((1, 10, 1000),       marks=pytest.mark.slow),       #31     # 4.4 Extreme ratios between height/width
            pytest.param((1, 9920, 1),        marks=pytest.mark.slow),       #32     # 4.4 Extreme ratios between height/width
            pytest.param((1, 10000, 1),       marks=pytest.mark.slow),       #33     # 4.4 Extreme ratios between height/width 
            pytest.param((1, 32, 64),         marks=pytest.mark.run_in_pp),  #34     # 4.1 Divisible by 32
            pytest.param((1, 160, 96),        marks=pytest.mark.slow),       #35     # 4.1 Divisible by 32
            pytest.param((1, 17, 41),         marks=pytest.mark.slow),       #36     # 4.2 Prime numbers
            pytest.param((1, 89, 3),          marks=pytest.mark.slow),       #37     # 4.2 Prime numbers

             # 3-dimensional shape, microbatch_size > 1:
            pytest.param((2, 3, 4),           marks=pytest.mark.run_in_pp),  #38     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((11, 45, 17),        marks=pytest.mark.slow),       #39     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((11, 1, 23),         marks=pytest.mark.slow),       #40     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((11, 64, 1),         marks=pytest.mark.slow),       #41     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((100, 100, 100),     marks=pytest.mark.slow),       #42     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((10, 1000, 100),     marks=pytest.mark.slow),       #43     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((10, 10000, 1),      marks=pytest.mark.slow),       #44     # 4.4 Extreme ratios between height/width
            pytest.param((32, 32, 64),        marks=pytest.mark.slow),       #45     # 4.1 Divisible by 32
            pytest.param((64, 160, 96),       marks=pytest.mark.slow),       #46     # 4.1 Divisible by 32
            pytest.param((11, 17, 41),        marks=pytest.mark.run_in_pp),  #47     # 4.2 Prime numbers
            pytest.param((13, 89, 3),         marks=pytest.mark.slow),       #48     # 4.2 Prime numbers

            # 4-dimensional shape, microbatch_size = 1:
            pytest.param((1, 2, 3, 4),        marks=pytest.mark.run_in_pp),  #49     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((1, 11, 45, 17),     marks=pytest.mark.slow),       #50     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((1, 11, 1, 23),      marks=pytest.mark.slow),       #51     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((1, 11, 64, 1),      marks=pytest.mark.slow),       #52     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((1, 100, 100, 100),  marks=pytest.mark.slow),       #53     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((1, 10, 1000, 100),  marks=pytest.mark.slow),       #54     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((1, 1, 10, 1000),    marks=pytest.mark.slow),       #55     # 4.4 Extreme ratios between height/width
            pytest.param((1, 1, 9920, 1),     marks=pytest.mark.slow),       #56     # 4.4 Extreme ratios between height/width
            pytest.param((1, 10, 10000, 1),   marks=pytest.mark.slow),       #57     # 4.4 Extreme ratios between height/width
            pytest.param((1, 32, 32, 64),     marks=pytest.mark.run_in_pp),  #58     # 4.1 Divisible by 32
            pytest.param((1, 64, 160, 96),    marks=pytest.mark.slow),       #59     # 4.1 Divisible by 32
            pytest.param((1, 11, 17, 41),     marks=pytest.mark.slow),       #60     # 4.2 Prime numbers
            pytest.param((1, 13, 89, 3),      marks=pytest.mark.slow),       #61     # 4.2 Prime numbers

            # 4-dimensional shape, microbatch_size > 1:
            pytest.param((3, 11, 45, 17),     marks=pytest.mark.run_in_pp),  #62     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((2, 2, 3, 4),        marks=pytest.mark.slow),       #63     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param((4, 11, 1, 23),      marks=pytest.mark.slow),       #64     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((5, 11, 64, 1),      marks=pytest.mark.slow),       #65     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param((6, 100, 100, 100),  marks=pytest.mark.slow),       #66     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((7, 10, 1000, 100),  marks=pytest.mark.slow),       #67     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param((8, 1, 10, 1000),    marks=pytest.mark.slow),       #68     # 4.4 Extreme ratios between height/width
            pytest.param((9, 1, 9920, 1),     marks=pytest.mark.slow),       #69     # 4.4 Extreme ratios between height/width
            pytest.param((10, 10, 10000, 1),  marks=pytest.mark.slow),       #70     # 4.4 Extreme ratios between height/width
            pytest.param((11, 32, 32, 64),    marks=pytest.mark.slow),       #71     # 4.1 Divisible by 32
            pytest.param((12, 64, 160, 96),   marks=pytest.mark.slow),       #72     # 4.1 Divisible by 32
            pytest.param((13, 11, 17, 41),    marks=pytest.mark.run_in_pp),  #73     # 4.2 Prime numbers
            pytest.param((14, 13, 89, 3),     marks=pytest.mark.slow),       #74     # 4.2 Prime numbers
    ]


def get_input_shapes_test_plan():
    return get_input_shapes()
    # return [shape for shape in get_input_shapes() if len(shape[0][0]) in (2, 3, 4,)]


def get_input_shapes_df_mf():
    return TestData.input_shapes_single
    # return [shape for shape in get_input_shapes() if len(shape[0][0]) in (4,)][:1]


test_plan = TestPlan(
    tests = [
        # Test plan: 
        # 2. Operand source(s):
        # 3. Operand shapes type(s):
        # 4. Operand / output size of dimensions
        TestVectors(
            operators=get_eltwise_binary_ops(),
            input_sources=TestData.INPUT_SOURCES,
            input_shapes=get_input_shapes_test_plan(),
        ),
        # Test plan: 
        # 5. Data format
        TestVectors(
            operators=get_eltwise_binary_ops(),
            input_sources=TestData.INPUT_SOURCES_SINGLE,
            input_shapes=get_input_shapes_df_mf(),
            dev_data_formats=TestData.dev_data_formats,
            math_fidelities=TestData.math_fidelities_defaults,
        ),
        # Test plan: 
        # 6. Math fidelity
        TestVectors(
            operators=get_eltwise_binary_ops(),
            input_sources=TestData.INPUT_SOURCES_SINGLE,
            input_shapes=get_input_shapes_df_mf(),
            dev_data_formats=TestData.dev_data_formats_defaults,
            math_fidelities=TestData.math_fidelities,
        ),
        # Unimplemented operators
        TestVectors(
            operators=get_not_implemented_pytorch_binary_ops(),
            input_sources=TestData.INPUT_SOURCES,
            input_shapes=TestData.input_shapes_single,
        ),
    ],
    failing_tests = [
        # Unsupported datatypes
        TestVectors(
            operators=None,
            input_sources=None,
            input_shapes=None,
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
        # PCC check fails for buggy shapes for all models
        TestVectors(
            operators=None,
            input_sources=None,
            input_shapes=[
                (1, 1),
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # PCC check fails for buggy shapes for model ModelConstEvalPass
        TestVectors(
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
        # PCC check fails for buggy shapes for div
        TestVectors(
            operators=["div", "divide", "true_divide", ],
            input_sources=[InputSource.FROM_HOST, InputSource.FROM_DRAM_QUEUE, ],
            input_shapes=[
                (1, 4),
                (1, 3),
                (3, 4),
                (1, 3, 4),
                (12, 64, 160, 96),
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # PCC check fails for buggy shapes for div
        TestVectors(
            operators=["div", "divide", "true_divide", ],
            input_sources=[InputSource.CONST_EVAL_PASS, ],
            input_shapes=[
                (1, 17),
                (45, 17),
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # PCC check fails for buggy shapes for ge
        TestVectors(
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
        # Not implemented operators
        TestVectors(
            operators=get_not_implemented_pytorch_binary_ops(),
            input_sources=TestData.INPUT_SOURCES,
            input_shapes=TestData.input_shapes_single,
            failing_reason=FailingReasons.NOT_IMPLEMENTED,
        ),
    ]
)


class BinaryTestParameterGenerator (TestParameterGenerator):

    def __init__(self):
        super(BinaryTestParameterGenerator, self).__init__()

        rng_limiter = random.Random(0)
        self.alpha_limiter = RateLimiter(rng_limiter, 100, 20)
        self.alpha_small_limiter = RateLimiter(rng_limiter, 100, 50)

    def produce(self, input_operator, input_source, input_shape, dev_data_format, math_fidelity, failing_result: Optional[TestResultFailing]):
            
        def get_failing_result(kwargs: Dict[str, Any]) -> Optional[FailingReasons]:

            # TODO check kwargs via test plan
            if "rounding_mode" in kwargs and kwargs["rounding_mode"] == "trunc" and input_source in (InputSource.CONST_EVAL_PASS,) and dev_data_format in (
                forge.DataFormat.Bfp2,  # PCC check failed for some shapes
                forge.DataFormat.Bfp2_b,
                forge.DataFormat.Bfp8,
                forge.DataFormat.Bfp8_b,  # PCC check failed for some shapes
                forge.DataFormat.Float32,
                forge.DataFormat.Int8,
                forge.DataFormat.Lf8,
                forge.DataFormat.RawUInt16,
            ):
                return TestResultFailing(FailingReasons.DATA_MISMATCH)

            if "alpha" in kwargs:
                if input_operator in ["add",]:
                    # 2D shapes that reduce to 1D are passing
                    if len(input_shape) == 2 and input_shape[-1] == 1:
                        return None
                    # TODO check kwargs range via test plan
                    if not 0.8 <= abs(kwargs['alpha']) <= 1.2:
                        # It looks like Forge is not supporting alpha parameter so PCC is always different
                        return TestResultFailing(FailingReasons.UNSUPPORTED_SPECIAL_CASE)

                return None

        kwargs_list = []  # collect list of kwargs to produce multiple test cases with different kwargs

        # TODO generate kwargs via test plan
        if input_operator in ["add", "sub", "substract"]:
            if self.alpha_limiter.is_allowed():
                if dev_data_format in TestData.dev_data_formats_int:
                    alpha_value = self.rng_params.randint(0, 100)
                elif self.alpha_small_limiter.is_allowed():
                    # small numbers
                    alpha_value = self.rng_params.uniform(-1.0, 1.0)
                else:
                    # regular number range
                    alpha_value = self.rng_params.uniform(5, 10000)
                    # support negative numbers
                    alpha_value *= self.rng_params.choice([-1, 1])
                kwargs = {
                    'alpha': alpha_value
                }
                kwargs_list.append(kwargs)
            kwargs_list.append({})
        elif input_operator in ["div", "divide"]:
            rounding_modes = ['trunc', 'floor', None]
            kwargs = {
                'rounding_mode': rounding_modes[self.rng_params.randint(0, 2)]
            }
            kwargs_list.append(kwargs)
        else:
            kwargs = {}
            kwargs_list.append(kwargs)

        params = []

        failing_result_original = failing_result

        for kwargs in kwargs_list:

            failing_result = failing_result_original

            # Check additional custom conditions for failing result
            if failing_result is None and kwargs is not None:
                failing_result = get_failing_result(kwargs)

            # These 10 operators are supported for CONST_EVAL_PASS
            if failing_result is not None and failing_result.failing_reason == FailingReasons.NOT_IMPLEMENTED and input_source in (InputSource.CONST_EVAL_PASS, ) and input_operator in (
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
            ):
                failing_result = None

            marks = self.get_marks(failing_result)

            param = pytest.param(input_operator, input_source, kwargs, input_shape, dev_data_format, math_fidelity, marks=marks, id=f"{input_operator}-{input_source.name}-{kwargs}-{input_shape}-{dev_data_format.name if dev_data_format else None}-{math_fidelity.name if math_fidelity else None}")
            params.append(param)
        
        return params


test_plan_generator = BinaryTestParameterGenerator()


@pytest.mark.parametrize("input_operator, input_source, kwargs, input_shape, dev_data_format, math_fidelity", test_plan_generator.generate(test_plan))
def test_pytorch_eltwise_binary_ops_per_test_plan(
    input_operator,
    input_source,
    kwargs,
    input_shape,
    test_device,
    dev_data_format,
    math_fidelity
):

    verify(
        test_device=test_device,
        input_source=input_source,
        input_operator=input_operator,
        input_shape=input_shape,
        kwargs=kwargs,
        dev_data_format=dev_data_format,
        math_fidelity=math_fidelity,
    )
# 1480 passed, 20 xfailed, 2 warnings in 529.46s (0:08:49) 
# 4 failed, 1352 passed, 212 xfailed, 115 xpassed, 2 warnings in 590.56s (0:09:50)


