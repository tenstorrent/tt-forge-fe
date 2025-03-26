# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from functools import reduce
import math
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
from test.operators.utils.test_data import TestCollectionTorch



class ModelFromAnotherOp(torch.nn.Module):

    model_name = "model_op_src_from_another_op"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelFromAnotherOp, self).__init__()
        self.testname = "Conv2d_pytorch_operator_" + opname + "_test_op_src_from_another_op"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = kwargs

        self.ct1 = self.operator(**self.kwargs)

    def forward(self, x: torch.Tensor):
        # we use Add operator to create one operands which is input for the Conv2d operator
        add = torch.add(x, x)
        output = self.ct1(add)
        return output


class ModelDirect(torch.nn.Module):

    model_name = "model_op_src_from_host"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelDirect, self).__init__()
        self.testname = "Conv2d_pytorch_operator_" + opname + "_test_op_src_from_host"
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
        self.testname = "Conv2d_pytorch_operator_" + opname + "_test_op_src_const_eval_pass"
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
            dev_data_format=test_vector.dev_data_format,
            math_fidelity=test_vector.math_fidelity,
            pcc=test_vector.pcc,
            warm_reset=warm_reset,
            deprecated_verification=False,
            verify_config=VerifyConfig(value_checker=AllCloseValueChecker(rtol=1e-2, atol=1e-2)),
            value_range=ValueRanges.SMALL,
            skip_forge_verification = False,
        )

@dataclass
class Conv2DParams:
    in_channels: int
    out_channels: int
    kernel_size: tuple
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    groups: int = 1
    bias: bool = True
    padding_mode: str = "zeros"

    max_kernel_height = 100
    max_kernel_width = 100

    @classmethod
    def to_kwargs(cls):
        return cls.__dict__

    def to_kwargs_list(cls):
        return [cls.__dict__]

    @classmethod
    def get_kernel_param(cls, rng, H_in, W_in, k_min_h = 1, k_min_w = 1, odd: bool = False):
        k_min_h = min(k_min_h, H_in)
        k_min_w = min(k_min_w, W_in)
        
        # Adjust minimum values to be odd if required
        if odd:
            if k_min_h % 2 == 0:
                k_min_h = max(k_min_h + 1, 1)
            if k_min_w % 2 == 0:
                k_min_w = max(k_min_w + 1, 1)
        
        # Initial limit based on input size
        k_h_limit = min(H_in // 3, cls.max_kernel_height)
        k_w_limit = min(W_in // 3, cls.max_kernel_width)
        
        # Ensure minimum size constraints
        k_h_limit = max(k_h_limit, k_min_h)
        k_w_limit = max(k_w_limit, k_min_w)
        
        # Generate kernel sizes within the computed limits
        k_h = rng.randint(k_min_h, k_h_limit)
        k_w = rng.randint(k_min_w, k_w_limit)
        
        # Ensure odd-sized kernels if required
        if odd:
            if k_h % 2 == 0:
                k_h = max(k_h - 1, k_min_h) if k_h - 1 >= k_min_h else k_h + 1
            if k_w % 2 == 0:
                k_w = max(k_w - 1, k_min_w) if k_w - 1 >= k_min_w else k_w + 1
        
        return k_h, k_w
    
    @classmethod
    def get_non_unit_stride_param(cls, rng, H_in, W_in):
        stride_h = rng.randint(2 if H_in > 1 else 1, H_in//5 if H_in//5 > 1 else 2)
        stride_w = rng.randint(2 if W_in > 1 else 1, W_in//5 if W_in//5 > 1 else 2)
        stride = (stride_h, stride_w)
        return stride

    @classmethod
    def get_groups_param(cls, in_channels, out_channels, mode="standard"):
        """
        Generates the `groups` parameter for a Conv2D operation.

        Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mode (str): Convolution mode - "standard", "depthwise", or "grouped".

        Returns:
        int: The appropriate groups value.
        """
        if mode == "standard":
            return 1  # Standard convolution (all input channels connect to all output channels)
        elif mode == "depthwise":
            return in_channels if in_channels == out_channels else 1  # Depthwise only valid if channels match
        elif mode == "grouped":
            return math.gcd(in_channels, out_channels)  # Largest common divisor for efficient grouping
        else:
            raise ValueError("Invalid mode. Choose from 'standard', 'depthwise', or 'grouped'.")
    
    @classmethod
    def find_k_d(cls, k_eff):
        results = []
        if k_eff > 100:
            k_eff = 100
        for k in range(1, k_eff + 1):  # Trying values from 1 to k_eff
            for d in range(2, k_eff + 1):
                if k + (k - 1) * (d - 1) == k_eff:
                    results.append((k, d))
        return results

class TestParamsData:

    __test__ = False  # Avoid collecting TestParamsData as a pytest test

    test_plan: TestPlan = None 

    @classmethod
    def generate_kwargs_all(cls, test_vector: TestVector, bias: bool = True):
        kwarg_list = []
        kwarg_list.extend(cls.generate_kwargs_no_zero_padding_unit_strides(test_vector, bias))   
        kwarg_list.extend(cls.generate_kwargs_zero_padding_unit_strides(test_vector, bias))
        kwarg_list.extend(cls.generate_kwargs_same_output_as_input(test_vector, bias))
        kwarg_list.extend(cls.generate_kwargs_output_bigger_then_input(test_vector, bias))
        kwarg_list.extend(cls.generate_kwargs_no_zero_padding_no_unit_strides(test_vector, bias))
        kwarg_list.extend(cls.generate_kwargs_zero_padding_no_unit_strides(test_vector, bias))
        kwarg_list.extend(cls.generate_kwargs_no_zero_padding_unit_strides_dilation(test_vector, bias))
        kwarg_list.extend(cls.generate_kwargs_zero_padding_unit_strides_dilation(test_vector, bias))
        kwarg_list.extend(cls.generate_kwargs_zero_padding_no_unit_strides_dilaiton(test_vector, bias))
        kwarg_list.extend(cls.generate_kwargs_groups_depthwise(test_vector, bias))
        kwarg_list.extend(cls.generate_kwargs_groups_grouped(test_vector, bias))
        return kwarg_list
    
    @classmethod
    def generate_kwargs_single(cls, test_vector: TestVector, bias: bool = True):
        kwarg_list = []
        kwarg_list.extend(cls.generate_kwargs_all_type_padding_unit_strides(test_vector, bias))
        return kwarg_list
    
    
    @classmethod
    def generate_kwargs_no_zero_padding_unit_strides(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # prepare params
        conv2DParams = Conv2DParams(
            in_channels=C_in,
            out_channels=rng.randint(1, C_in + 10),  # it can be less, equal or greater than in_channels
            kernel_size=Conv2DParams.get_kernel_param(rng, H_in, W_in),
            bias=bias
        )
        # create kwargs
        return conv2DParams.to_kwargs_list()
    
    @classmethod
    def generate_kwargs_zero_padding_unit_strides(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # prepare params
        conv2DParams = Conv2DParams(
            in_channels=C_in,
            out_channels=rng.randint(1, C_in + 10),  # it can be less, equal or greater than in_channels
            kernel_size=Conv2DParams.get_kernel_param(rng, H_in, W_in),
            padding=rng.randint(1, 4),
            bias=bias
        )
        return conv2DParams.to_kwargs_list()
    
    @classmethod
    def generate_kwargs_same_output_as_input(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # Half (same) padding
        # o = (i - k) + 2p + 1 = i - k + 2p + 1
        # o = i 
        # => -k + 2p + 1 = 0
        # => p = (k - 1) / 2, k <> 1
        # prepare params
        kernel_size = Conv2DParams.get_kernel_param(rng, H_in, W_in, k_min_h=2, k_min_w=2, odd=True)
        if(kernel_size[0] == 1 or kernel_size[1] == 1):
            return []
        padding_h = ((kernel_size[0] - 1) if (kernel_size[0] - 1) > 0 else 1) // 2
        padding_w = ((kernel_size[1] - 1) if (kernel_size[1] - 1) > 0 else 1) // 2
        padding = (padding_h, padding_w)
        conv2DParams = Conv2DParams(
            in_channels = C_in,
            out_channels = rng.randint(1, C_in + 10),  # it can be less, equal or greater than in_channels
            kernel_size = kernel_size,
            padding = padding,
            bias = bias
        )
        return conv2DParams.to_kwargs_list()
    
    @classmethod
    def generate_kwargs_output_bigger_then_input(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # Full padding
        # o = (i - k) + 2p + 1 = i - k + 2p + 1
        # p = k − 1
        # => o = i - k + 2k - 2 + 1
        # => o = i + k - 1
        # o > i, k <> 1
        # prepare params
        kernel_size = Conv2DParams.get_kernel_param(rng, H_in, W_in, k_min_h=2, k_min_w=2)
        if(kernel_size[0] == 1 and kernel_size[1] == 1):
            return []
        padding_h = kernel_size[0] - 1
        padding_w = kernel_size[1] - 1
        padding = (padding_h, padding_w)
        conv2DParams = Conv2DParams(
            in_channels = C_in,
            out_channels = rng.randint(1, C_in + 10),  # it can be less, equal or greater than in_channels
            kernel_size = kernel_size,
            padding = padding,
            bias = bias
        )
        return conv2DParams.to_kwargs_list()

    @classmethod
    def generate_kwargs_no_zero_padding_no_unit_strides(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # prepare params
        conv2DParams = Conv2DParams(
            in_channels = C_in,
            out_channels = rng.randint(1, C_in + 10),  # it can be less, equal or greater than in_channels
            kernel_size = Conv2DParams.get_kernel_param(rng, H_in, W_in),
            stride = Conv2DParams.get_non_unit_stride_param(rng, H_in, W_in),
            bias = bias
        )
        return conv2DParams.to_kwargs_list()

    @classmethod
    def generate_kwargs_zero_padding_no_unit_strides(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # prepare params
        conv2DParams = Conv2DParams(
            in_channels = C_in,
            out_channels = rng.randint(1, C_in + 10),  # it can be less, equal or greater than in_channels
            kernel_size = Conv2DParams.get_kernel_param(rng, H_in, W_in),
            stride = Conv2DParams.get_non_unit_stride_param(rng, H_in, W_in),
            padding = rng.randint(1, 4),
            bias=bias
        )
        return conv2DParams.to_kwargs_list()

    @classmethod
    def generate_kwargs_all_type_padding_unit_strides(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # prepare params
        out_channels = rng.randint(1, C_in + 10)  # it can be less, equal or greater than in_channels
        kernel_size = Conv2DParams.get_kernel_param(rng, H_in, W_in)
        padding_list =[]
        padding_list.append(rng.randint(1, 15))
        padding_list.append("valid")
        padding_list.append("same")
        padding_mode_list = [
            "zeros", # already tested in previous cases but we test it again because of valid and same padding
            "reflect",
            "replicate", 
            "circular"
        ]
        # create kwargs
        kwarg_list = []
        for padding in padding_list:
            for padding_mode in padding_mode_list:
                conv2DParams = Conv2DParams(
                    in_channels = C_in,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                    bias=bias
                )
                kwarg_list.append(conv2DParams.__dict__)
        return kwarg_list
    
    @classmethod
    def generate_kwargs_no_zero_padding_unit_strides_dilation(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # prepare params
        # k_eff = k + (k − 1)(d − 1).
        k_eff_h = rng.randint(H_in//2 if H_in//2 > 0 else 1, H_in)
        k_eff_w = rng.randint(W_in//2 if W_in//2 > 0 else 1, W_in)
        possible_k_d_h = Conv2DParams.find_k_d(k_eff_h)
        possible_k_d_w = Conv2DParams.find_k_d(k_eff_w)
        if not possible_k_d_h or not possible_k_d_w:
            return []
        k_d_h = rng.choice(possible_k_d_h)
        k_d_w = rng.choice(possible_k_d_w)
        kernel_size_h = k_d_h[0]
        kernel_size_w = k_d_w[0]
        kernel_size = (kernel_size_h, kernel_size_w)
        dilation_h = k_d_h[1]
        dilation_w = k_d_w[1]
        dilation = (dilation_h, dilation_w)
        conv2DParams = Conv2DParams(
            in_channels = C_in,
            out_channels = rng.randint(1, C_in + 10),  # it can be less, equal or greater than in_channels
            kernel_size = kernel_size,
            dilation=dilation,
            bias=bias
        )
        return conv2DParams.to_kwargs_list()
    
    @classmethod
    def generate_kwargs_zero_padding_unit_strides_dilation(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # prepare params
        # k_eff = k + (k − 1)(d − 1).
        k_eff_h = rng.randint(H_in//2 if H_in//2 > 0 else 1, H_in)
        k_eff_w = rng.randint(W_in//2 if W_in//2 > 0 else 1, W_in)
        possible_k_d_h = Conv2DParams.find_k_d(k_eff_h)
        possible_k_d_w = Conv2DParams.find_k_d(k_eff_w)
        if not possible_k_d_h or not possible_k_d_w:
            return []
        k_d_h = rng.choice(possible_k_d_h)
        k_d_w = rng.choice(possible_k_d_w)
        kernel_size_h = k_d_h[0]
        kernel_size_w = k_d_w[0]
        kernel_size = (kernel_size_h, kernel_size_w)
        dilation_h = k_d_h[1]
        dilation_w = k_d_w[1]
        dilation = (dilation_h, dilation_w)
        conv2DParams = Conv2DParams(
            in_channels = C_in,
            out_channels = rng.randint(1, C_in + 10),  # it can be less, equal or greater than in_channels
            kernel_size = kernel_size,
            dilation=dilation,
            padding = rng.randint(1, 4),
            bias=bias
        )
        return conv2DParams.to_kwargs_list()
    
    @classmethod
    def generate_kwargs_zero_padding_no_unit_strides_dilaiton(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # prepare params
        # k_eff = k + (k − 1)(d − 1).
        k_eff_h = rng.randint(H_in//2 if H_in//2 > 0 else 1, H_in)
        k_eff_w = rng.randint(W_in//2 if W_in//2 > 0 else 1, W_in)
        possible_k_d_h = Conv2DParams.find_k_d(k_eff_h)
        possible_k_d_w = Conv2DParams.find_k_d(k_eff_w)
        if not possible_k_d_h or not possible_k_d_w:
            return []
        k_d_h = rng.choice(possible_k_d_h)
        k_d_w = rng.choice(possible_k_d_w)
        kernel_size_h = k_d_h[0]
        kernel_size_w = k_d_w[0]
        kernel_size = (kernel_size_h, kernel_size_w)
        dilation_h = k_d_h[1]
        dilation_w = k_d_w[1]
        dilation = (dilation_h, dilation_w)
        conv2DParams = Conv2DParams(
            in_channels = C_in,
            out_channels = rng.randint(1, C_in + 10),  # it can be less, equal or greater than in_channels
            kernel_size = kernel_size,
            dilation=dilation,
            stride = Conv2DParams.get_non_unit_stride_param(rng, H_in, W_in),
            padding = rng.randint(1, 4),
            bias=bias
        )
        return conv2DParams.to_kwargs_list()

    @classmethod
    def generate_kwargs_groups_depthwise(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # prepare params
        groups = Conv2DParams.get_groups_param(C_in, C_in, mode="depthwise") 
        if groups == 1:
            return [] # skip: already tested in previous cases
        conv2DParams = Conv2DParams(
            in_channels = C_in,
            out_channels = C_in,
            kernel_size = Conv2DParams.get_kernel_param(rng, H_in, W_in),
            groups = groups,
            bias = bias
        )
        return conv2DParams.to_kwargs_list()

    @classmethod
    def generate_kwargs_groups_grouped(cls, test_vector: TestVector, bias: bool = True):
        rng = random.Random(sum(test_vector.input_shape))
        N, C_in, H_in, W_in = test_vector.input_shape
        # prepare params
        C_out = rng.randint(1, C_in + 10)
        while C_out == C_in:
            C_out = rng.randint(1, C_in + 10)
        groups = Conv2DParams.get_groups_param(C_in, C_out, mode="grouped")
        if groups == 1 or groups == C_in:
            return []
        conv2DParams = Conv2DParams(
            in_channels = C_in,
            out_channels = C_out,
            kernel_size = Conv2DParams.get_kernel_param(rng, H_in, W_in),
            groups = groups,
            bias = bias
        )
        return conv2DParams.to_kwargs_list()

class TestCollectionData:

    __test__ = False  # Avoid collecting TestCollectionData as a pytest test

    all = TestCollection(
        operators=[
            "Conv2d",  # 00
        ],
        input_sources=TestCollectionCommon.all.input_sources,
        # only 4D input tensors are supported
        input_shapes=[input_shape for input_shape in TestCollectionCommon.all.input_shapes if len(input_shape) == 4],
        dev_data_formats=TestCollectionCommon.all.dev_data_formats,
        math_fidelities=TestCollectionCommon.all.math_fidelities,
    )

    single = TestCollection(
        input_sources=TestCollectionCommon.single.input_sources,
        input_shapes=[
            (3, 11, 45, 17),
        ],
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
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_no_zero_padding_unit_strides(test_vector),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_zero_padding_unit_strides(test_vector),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_same_output_as_input(test_vector),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_output_bigger_then_input(test_vector),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_no_zero_padding_no_unit_strides(test_vector),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_zero_padding_no_unit_strides(test_vector),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_all_type_padding_unit_strides(test_vector),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_zero_padding_unit_strides_dilation(test_vector),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_zero_padding_no_unit_strides_dilaiton(test_vector),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_no_zero_padding_unit_strides_dilation(test_vector),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_groups_depthwise(test_vector),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_groups_grouped(test_vector),
        ),

        # All tests but with no bias
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_all(test_vector, bias=False),
        ),
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_single(test_vector, bias=False),
        ),

        # Using bias=False for data format and math fidelity tests because its more stable at the moment
        # Test Data formats collection:
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_no_zero_padding_unit_strides(test_vector, bias=False),
            dev_data_formats=[
                item
                for item in TestCollectionTorch.all.dev_data_formats
                if item not in TestCollectionTorch.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
        ),
        # Test math fidelity collection:
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_no_zero_padding_unit_strides(test_vector, bias=False),
            # dev_data_formats=TestCollectionTorch.single.dev_data_formats,  # Can't use it because it's unsupported data format
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
    ],
    failing_rules=[
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [
        TestParamsData.test_plan,
    ]
