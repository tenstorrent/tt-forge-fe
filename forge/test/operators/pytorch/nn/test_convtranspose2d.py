# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from functools import reduce
import random
import pytest

from typing import List, Dict, Type, Optional, Any
from loguru import logger

import torch
import forge
import forge.op

from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AllCloseValueChecker

from test.operators.utils import InputSourceFlags, VerifyUtils, ValueRanges
from test.operators.utils import InputSource
from test.operators.utils import TestVector
from test.operators.utils import TestPlan
from test.operators.utils import FailingReasons
from test.operators.utils.compat import TestDevice
from test.operators.utils.compat import TestTensorsUtils
from test.operators.utils import TestCollection
from test.operators.utils import TestCollectionCommon


class ModelFromAnotherOp(torch.nn.Module):

    model_name = "model_op_src_from_another_op"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelFromAnotherOp, self).__init__()
        self.testname = "ConvTranspose2d_pytorch_operator_" + opname + "_test_op_src_from_another_op"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

        self.ct1 = self.operator(**self.kwargs)

    def forward(self, x: torch.Tensor):
        # we use Add operator to create one operands which is input for the ConvTranspose2d operator
        add = torch.add(x, x)
        output = self.ct1(add)
        return output


class ModelDirect(torch.nn.Module):

    model_name = "model_op_src_from_host"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelDirect, self).__init__()
        self.testname = "ConvTranspose2d_pytorch_operator_" + opname + "_test_op_src_from_host"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

        self.ct1 = self.operator(**self.kwargs)

    def forward(self, x: torch.Tensor):
        output = self.ct1(x)
        return output


class ModelConstEvalPass(torch.nn.Module):

    model_name = "model_op_src_const_eval_pass"

    def __init__(self, operator, opname, shape, kwargs, dtype):
        super(ModelConstEvalPass, self).__init__()
        self.testname = "ConvTranspose2d_pytorch_operator_" + opname + "_test_op_src_const_eval_pass"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

        self.constant = torch.rand(self.shape, dtype=dtype)
        self.ct1 = self.operator(**self.kwargs)

    def forward(self, x: torch.Tensor):
        v1 = self.ct1(self.constant)
        # v2 = torch.add(x, x)
        v2 = self.ct1(x)
        # add consume inputs
        add = torch.add(v1, v2)
        return add


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
        input_params: List[Dict] = [],
        number_of_operands: int = 1,
        warm_reset: bool = False,
    ):
        """Common verification function for all tests"""

        input_source_flag: InputSourceFlags = None
        if test_vector.input_source in (InputSource.FROM_DRAM_QUEUE,):
            input_source_flag = InputSourceFlags.FROM_DRAM

        operator = getattr(torch.nn, test_vector.operator)

        kwargs = test_vector.kwargs if test_vector.kwargs else {}

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        if test_vector.input_source == InputSource.CONST_EVAL_PASS:
            pytorch_model = model_type(
                operator=operator,
                opname=test_vector.operator,
                shape=test_vector.input_shape,
                kwargs=kwargs,
                dtype=TestTensorsUtils.get_dtype_for_df(test_vector.dev_data_format),
            )
        else:
            pytorch_model = model_type(
                operator=operator,
                opname=test_vector.operator,
                shape=test_vector.input_shape,
                kwargs=kwargs,
            )

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
            warm_reset=warm_reset,
            deprecated_verification=False,
            verify_config=VerifyConfig(value_checker=AllCloseValueChecker(rtol=1e-2, atol=1e-2)),
            value_range=ValueRanges.SMALL,
            skip_forge_verification = True,
        )


class TestParamsData:

    __test__ = False  # Avoid collecting TestParamsData as a pytest test

    test_plan: TestPlan = None

    @classmethod
    def calculate_param_dilation(cls, rng, N: int, C_in: int, H_in: int, W_in: int):
        dilations = []
        # create as int 
        dilation = rng.randint(1, 3)
        dilations.append(dilation)
        # create as tuple
        dilation = tuple([rng.randint(1, 3), rng.randint(1, 3)])
        dilations.append(dilation)
        return dilations

    @classmethod
    def calculate_param_stride(cls, rng, N: int, C_in: int, H_in: int, W_in: int):
        strides = []
        # create as int
        stride = rng.randint(1, 2)
        strides.append(stride)
        # create as tuple
        stride = tuple([rng.randint(1, 2), rng.randint(1, 2)])
        strides.append(stride)
        return strides
    
    @classmethod
    def calculate_param_kernel(cls, rng, N, C_in, H_in, W_in, dilation_params):
        kernels = []
        dilation = 1
        for item in dilation_params:
            if (not isinstance(item, tuple)):
                dilation = item
                break
        # there is no point testing values outside this range [3,7]
        kernel_size_min_test_size = 3 
        kernel_size_max_test_size = 7
        # assert that kernel value will fit in the input shape
        # if isinstance(kernel_size, tuple):
        #     assert dilation * (kernel_size[0] - 1) < H_in, "Invalid kernel height!"
        #     assert dilation * (kernel_size[1] - 1) < W_in, "Invalid kernel width!"
        # else:
        #     assert dilation * (kernel_size - 1) < H_in, "Invalid height"
        #     assert dilation * (kernel_size - 1) < W_in, "Invalid width"

        k_maxh = max(kernel_size_min_test_size, int(H_in / dilation + 1) % kernel_size_max_test_size) 
        k_maxw = max(kernel_size_min_test_size, int(W_in / dilation + 1) % kernel_size_max_test_size) 

        # Two cases for kernel size
        # 1. kernel is equal to integer
        kernel_size_option1 = rng.randint(kernel_size_min_test_size, k_maxh)
        kernel_size_option1_odd = kernel_size_option1 if kernel_size_option1 % 2 != 0 else kernel_size_option1 + 1
        kernel_size_option2 = rng.randint(kernel_size_min_test_size, k_maxw)
        kernel_size_option2_odd = kernel_size_option2 if kernel_size_option2 % 2 != 0 else kernel_size_option2 + 1
        kernel_size_int = random.choice([kernel_size_option1, kernel_size_option1_odd, kernel_size_option2, kernel_size_option2_odd])
        kernels.append(kernel_size_int)
        # 2. kernel is equal to tuple
        kernel_size_tuple = (kernel_size_option1, kernel_size_option2)
        kernels.append(kernel_size_tuple)
        return kernels

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):
        kwarg_list = []
        rng = random.Random(sum(test_vector.input_shape))
        N = test_vector.input_shape[0]
        C_in = test_vector.input_shape[-3]
        H_in = test_vector.input_shape[-2]
        W_in = test_vector.input_shape[-1]

        in_channels = C_in
        out_channels = rng.randint(1, C_in + 100)  # it can be less, equal or greater than in_channels

        dilation_params = cls.calculate_param_dilation(rng, N, C_in, H_in, W_in) 
        kernel_params = cls.calculate_param_kernel(rng, N, C_in, H_in, W_in, dilation_params)
        stride_params = cls.calculate_param_stride(rng, N, C_in, H_in, W_in)
        
        for kernel_size in kernel_params:
            for dilation in dilation_params:
                for stride in stride_params:
                    kwargs = {
                        "in_channels": in_channels,
                        "out_channels": out_channels,
                        "kernel_size": kernel_size,
                        "stride": stride,
                        "padding": rng.randint(0, 2),
                        # "output_padding": rng.randint(0, 2),
                        "dilation": dilation,
                        # "groups": rng.randint(1, 3),
                        "bias": rng.choice([True, False]),
                    }
                    kwarg_list.append(kwargs)            
        
        return kwarg_list


class TestCollectionData:

    __test__ = False  # Avoid collecting TestCollectionData as a pytest test

    all = TestCollection(
        operators=[
            "ConvTranspose2d",  # 00
        ],
        input_sources=TestCollectionCommon.all.input_sources,
        # only 4D input tensors are supported
        input_shapes=[input_shape for input_shape in TestCollectionCommon.all.input_shapes if len(input_shape) == 4],
        dev_data_formats=TestCollectionCommon.all.dev_data_formats,
        math_fidelities=TestCollectionCommon.all.math_fidelities,
    )

    single = TestCollection(
        input_sources=TestCollectionCommon.single.input_sources,
        input_shapes=TestCollectionCommon.single.input_shapes,
        dev_data_formats=TestCollectionCommon.single.dev_data_formats,
        math_fidelities=TestCollectionCommon.single.math_fidelities,
    )


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
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
            dev_data_formats=TestCollectionCommon.float.dev_data_formats,
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
            math_fidelities=TestCollectionData.all.math_fidelities,
        ),
    ],
    failing_rules=[],
)


def get_test_plans() -> List[TestPlan]:
    return [
        TestParamsData.test_plan,
    ]
