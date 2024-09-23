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

from typing import List, Dict, Type, Optional
from loguru import logger

import random
import torch
import forge
import forge.op

from forge.op_repo import TensorShape

from test.operators.utils import InputSourceFlags, VerifyUtils
from test.operators.utils import ShapeUtils
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


class ModelFromHost(torch.nn.Module):

    model_name = "model_op_src_from_host"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelFromHost, self).__init__()
        self.testname = "Element_wise_pytorch_binary_operator_" + opname + "_test_op_src_from_host"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        output = self.operator(x, y, **self.kwargs)
        return output


class ModelFromDramQueue(torch.nn.Module):

    model_name = "model_op_src_from_dram_queue"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelFromDramQueue, self).__init__()
        self.testname = "Element_wise_pytorch_binary_operator_" + opname + "_test_op_src_from_dram_queue"
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



def verify(
    test_device: TestDevice,
    model_type: Type[torch.nn.Module],
    input_operator: str,
    input_shape: TensorShape,
    number_of_operands: int,
    kwargs: Dict = {},
    input_params: List[Dict] = [],
    input_source_flag: InputSourceFlags = None,
    dev_data_format: forge.DataFormat = None,
    math_fidelity: forge.MathFidelity = None,
    check_pcc: Optional[bool] = True, 
    pcc: Optional[float] = None,
):
    '''Common verification function for all tests'''

    xfail_test(input_operator, input_shape, model_type)

    operator = getattr(torch, input_operator)

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
        check_pcc=check_pcc,
        pcc=pcc,
    )


MODEL_TYPES = [
    ModelFromAnotherOp,
    ModelFromHost,
    ModelFromDramQueue,
    ModelConstEvalPass
]

def xfail_test(
    input_operator: str,
    input_shape: TensorShape,
    model_type: Type[torch.nn.Module],
):
    s = get_input_shapes()

    match input_operator:
        case "add" | "div" | "divide" | "mul" | "multiply" | "true_divide" | "ge" | "greater_equal" | "sub":
            if(input_shape == s[3][0][0]): 
                # E         AssertionError: PCC for single values doesn't work
                pytest.xfail(reason=FailingReasons.BUGGY_SHAPE)



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

@pytest.mark.parametrize("input_operator", get_eltwise_binary_ops())
@pytest.mark.parametrize("model_type", MODEL_TYPES)
@pytest.mark.parametrize("input_shape", ShapeUtils.extend_shapes_with_id(get_input_shapes()))
def test_pytorch_eltwise_binary_ops_per_test_plan(
    input_operator,
    model_type,
    input_shape,
    test_device,
    dev_data_format=None, 
    input_math_fidelity=None
):
    
    input_source_flag = None
    if model_type == ModelFromDramQueue:
        input_source_flag = InputSourceFlags.FROM_DRAM

    kwargs = {}
    if input_operator in ["add", "sub", "substract"] and kwargs_limiter.is_allowed():
        kwargs['alpha'] = random.uniform(0.5, 1000)
        # It looks like Forge is not supporting alpha parameter so PCC is always different
        pytest.xfail(reason=FailingReasons.UNSUPPORTED_SPECIAL_CASE)
    elif input_operator in ["div", "divide"]:
        rounding_modes = ['trunc', 'floor', None]
        kwargs['rounding_mode'] = rounding_modes[random.randint(0, 2)]


    verify(
        test_device=test_device,
        model_type=model_type,
        input_operator=input_operator,
        input_shape=input_shape,
        number_of_operands=2,
        kwargs=kwargs,
        input_source_flag=input_source_flag,
        dev_data_format=dev_data_format,
        math_fidelity=input_math_fidelity,
    )
# 1480 passed, 20 xfailed, 2 warnings in 529.46s (0:08:49) 

rng_limiter = random.Random(0)
kwargs_limiter = RateLimiter(rng_limiter, 100, 50)


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

input_shapes=[
    (1, 2, 3, 4),
]


@pytest.mark.parametrize("input_operator", get_not_implemented_pytorch_binary_ops())
@pytest.mark.parametrize("model_type", MODEL_TYPES)
@pytest.mark.parametrize("input_shape", input_shapes)
@pytest.mark.xfail(reason=FailingReasons.NOT_IMPLEMENTED)
def test_not_implemented_pytorch_eltwise_binary_ops_per_test_plan(
    input_operator,
    model_type,
    input_shape,
    test_device,
    dev_data_format=None, 
    input_math_fidelity=None
):

    verify(
        test_device=test_device,
        model_type=model_type,
        input_operator=input_operator,
        input_shape=input_shape,
        number_of_operands=2,
        dev_data_format=dev_data_format,
        math_fidelity=input_math_fidelity,
    )
# 78 xfailed, 10 xpassed, 2 warnings in 30.13s
# Those 10 tests that are xpassed are for operators: floor_divide, fmod, remainder, eq, ne, le, gt, lt, maximum, minimum which are working only with model ModelConstEvalPass. 
# They are not working with other models, as they are not implemented so we can't test them yet.



########## TEST DATA FORMAT AND MATH FIDELITY FOR ALL IMPLEMENTED BINARY OPS

# We will not test all combinations of Data Format and Math Fidelity because it would be too much tests. 
#   1. First we will choose Data Format to be Float16_b and test all Math Fidelity values
#   2. Then we will set Math Fidelity to HiFi4 and test all Data Formats. 

def get_input_shape():
    return  (1, 45, 17)     #0     # 3.1 Full tensor (i.e. full expected shape)

dev_data_formats = [
    forge.DataFormat.Float16_b,
]

compiler_math_fidelity = [
    forge.MathFidelity.LoFi,
    forge.MathFidelity.HiFi2,
    forge.MathFidelity.HiFi3,
    forge.MathFidelity.HiFi4,
]

@pytest.mark.parametrize("input_operator", get_eltwise_binary_ops())
@pytest.mark.parametrize("dev_data_format", dev_data_formats)
@pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
def test_pytorch_eltwise_binary_ops_mf_inputs(input_operator, test_device, dev_data_format, math_fidelity):
    test_pytorch_eltwise_binary_ops_per_test_plan(
        input_operator,
        ModelFromHost,
        get_input_shape(),
        test_device,
        dev_data_format, 
        math_fidelity
    )


        
dev_data_formats=[
    pytest.param(forge.DataFormat.Bfp2, id="Bfp2"),
    pytest.param(forge.DataFormat.Bfp2_b, id="Bfp2_b"),
    pytest.param(forge.DataFormat.Bfp4, id="Bfp4"),
    pytest.param(forge.DataFormat.Bfp4_b, id="Bfp4_b"),
    pytest.param(forge.DataFormat.Bfp8, id="Bfp8"),
    pytest.param(forge.DataFormat.Bfp8_b, id="Bfp8_b"),

    pytest.param(forge.DataFormat.Float16, id="Float16"),
    pytest.param(forge.DataFormat.Float16_b, id="Float16_b"),
    pytest.param(forge.DataFormat.Float32, id="Float32"),
    pytest.param(forge.DataFormat.Int8, id="Int8"),

    pytest.param(forge.DataFormat.Lf8, id="Lf8"),
    pytest.param(forge.DataFormat.RawUInt16, id="RawUInt16"),
    pytest.param(forge.DataFormat.RawUInt32, id="RawUInt32"),
    pytest.param(forge.DataFormat.RawUInt8, id="RawUInt8"),
    pytest.param(forge.DataFormat.UInt16, id="UInt16"),
]

compiler_math_fidelity = [
    forge.MathFidelity.HiFi4,
]

@pytest.mark.parametrize("input_operator", get_eltwise_binary_ops())
@pytest.mark.parametrize("model_type", [ModelFromHost])
@pytest.mark.parametrize("input_shape", [get_input_shape()])
@pytest.mark.parametrize("dev_data_format", dev_data_formats)
@pytest.mark.parametrize("math_fidelity", compiler_math_fidelity)
def test_pytorch_eltwise_binary_ops_df_inputs(input_operator, model_type, input_shape, test_device, dev_data_format, math_fidelity):
    test_pytorch_eltwise_binary_ops_per_test_plan(
        input_operator,
        model_type,
        input_shape,
        test_device,
        dev_data_format, 
        math_fidelity
    )



