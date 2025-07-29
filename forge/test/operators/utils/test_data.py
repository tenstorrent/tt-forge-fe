# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Test data for various test cases

import pytest
import forge
import torch

from forge import MathFidelity, DataFormat

from . import InputSource
from . import TestCollection


class TestCollectionCommon:
    """
    Shared test collection for all operators. Defined here to avoid duplication of test data.
    Contains a single test case and a collection of all test cases.
    """

    __test__ = False  # Avoid collecting TestCollectionCommon as a pytest test

    single = TestCollection(
        input_sources=[
            InputSource.FROM_HOST,
        ],
        input_shapes=[
            (1, 2, 3, 4),
        ],
        dev_data_formats=[
            forge.DataFormat.Float16_b,
        ],
        math_fidelities=[
            forge.MathFidelity.HiFi4,
        ],
    )

    all = TestCollection(
        operators=None,
        input_sources=[
            InputSource.FROM_ANOTHER_OP,
            InputSource.FROM_HOST,
            InputSource.CONST_EVAL_PASS,
        ],
        input_shapes=[
            # 2-dimensional shape, microbatch_size = 1:
            pytest.param(
                (1, 4), marks=pytest.mark.run_in_pp
            ),  #            #00      # 3.1 Full tensor (i.e. full expected shape)
            pytest.param(
                (1, 17), marks=pytest.mark.slow
            ),  #                #01      # 3.1 Full tensor (i.e. full expected shape)
            pytest.param(
                (1, 23), marks=pytest.mark.slow
            ),  #                #02      # 3.2 Tensor reduce on one or more dims to 1
            pytest.param(
                (1, 1), marks=pytest.mark.slow
            ),  #                 #03      # 3.2 Tensor reduce on one or more dims to 1
            pytest.param(
                (1, 100), marks=pytest.mark.slow
            ),  #               #04      # 4.3 Very large (thousands, 10s of thousands)
            pytest.param(
                (1, 500), marks=pytest.mark.slow
            ),  #               #05      # 4.3 Very large (thousands, 10s of thousands)
            pytest.param(
                (1, 1000), marks=pytest.mark.slow
            ),  #              #06      # 4.4 Extreme ratios between height/width
            pytest.param(
                (1, 1920), marks=pytest.mark.slow
            ),  #              #07      # 4.4 Extreme ratios between height/width
            pytest.param(
                (1, 10000), marks=pytest.mark.slow
            ),  #             #08      # 4.4 Extreme ratios between height/width
            pytest.param((1, 64), marks=pytest.mark.run_in_pp),  #           #09      # 4.1 Divisible by 32
            pytest.param((1, 96), marks=pytest.mark.slow),  #                #10      # 4.1 Divisible by 32
            pytest.param((1, 41), marks=pytest.mark.slow),  #                #11      # 4.2 Prime numbers
            pytest.param((1, 3), marks=pytest.mark.slow),  #                 #12      # 4.2 Prime numbers
            # 2-dimensional shape, microbatch_size > 1:
            pytest.param(
                (3, 4), marks=pytest.mark.run_in_pp
            ),  #            #13      # 3.1 Full tensor (i.e. full expected shape)
            pytest.param(
                (45, 17), marks=pytest.mark.slow
            ),  #               #14      # 3.1 Full tensor (i.e. full expected shape)
            pytest.param(
                (64, 1), marks=pytest.mark.slow
            ),  #                #15      # 3.2 Tensor reduce on one or more dims to 1
            pytest.param(
                (100, 100), marks=pytest.mark.slow
            ),  #             #16      # 4.3 Very large (thousands, 10s of thousands)
            pytest.param(
                (1000, 100), marks=pytest.mark.slow
            ),  #            #17      # 4.3 Very large (thousands, 10s of thousands)
            pytest.param(
                (10, 1000), marks=pytest.mark.slow
            ),  #             #18      # 4.4 Extreme ratios between height/width
            pytest.param(
                (9920, 1), marks=pytest.mark.slow
            ),  #              #19      # 4.4 Extreme ratios between height/width
            pytest.param(
                (10000, 1), marks=pytest.mark.slow
            ),  #             #20      # 4.4 Extreme ratios between height/width
            pytest.param((32, 64), marks=pytest.mark.slow),  #               #21      # 4.1 Divisible by 32
            pytest.param((160, 96), marks=pytest.mark.slow),  #              #22      # 4.1 Divisible by 32
            pytest.param((17, 41), marks=pytest.mark.run_in_pp),  #          #23      # 4.2 Prime numbers
            pytest.param((89, 3), marks=pytest.mark.slow),  #                #24      # 4.2 Prime numbers
            # 3-dimensional shape, microbatch_size = 1:
            pytest.param(
                (1, 3, 4), marks=pytest.mark.run_in_pp
            ),  #         #25     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param(
                (1, 45, 17), marks=pytest.mark.slow
            ),  #            #26     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param(
                (1, 1, 23), marks=pytest.mark.slow
            ),  #             #27     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param(
                (1, 64, 1), marks=pytest.mark.slow
            ),  #             #28     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param(
                (1, 100, 100), marks=pytest.mark.slow
            ),  #          #29     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param(
                (1, 1000, 100), marks=pytest.mark.slow
            ),  #         #30     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param(
                (1, 10, 1000), marks=pytest.mark.slow
            ),  #          #31     # 4.4 Extreme ratios between height/width
            pytest.param(
                (1, 9920, 1), marks=pytest.mark.slow
            ),  #           #32     # 4.4 Extreme ratios between height/width
            pytest.param(
                (1, 10000, 1), marks=pytest.mark.slow
            ),  #          #33     # 4.4 Extreme ratios between height/width
            pytest.param((1, 32, 64), marks=pytest.mark.run_in_pp),  #       #34     # 4.1 Divisible by 32
            pytest.param((1, 160, 96), marks=pytest.mark.slow),  #           #35     # 4.1 Divisible by 32
            pytest.param((1, 17, 41), marks=pytest.mark.slow),  #            #36     # 4.2 Prime numbers
            pytest.param((1, 89, 3), marks=pytest.mark.slow),  #             #37     # 4.2 Prime numbers
            # 3-dimensional shape, microbatch_size > 1:
            pytest.param(
                (2, 3, 4), marks=pytest.mark.run_in_pp
            ),  #         #38     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param(
                (11, 45, 17), marks=pytest.mark.slow
            ),  #           #39     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param(
                (11, 1, 23), marks=pytest.mark.slow
            ),  #            #40     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param(
                (11, 64, 1), marks=pytest.mark.slow
            ),  #            #41     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param(
                (100, 100, 100), marks=pytest.mark.slow
            ),  #        #42     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param(
                (10, 1000, 100), marks=pytest.mark.slow
            ),  #        #43     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param(
                (10, 10000, 1), marks=pytest.mark.slow
            ),  #         #44     # 4.4 Extreme ratios between height/width
            pytest.param((32, 32, 64), marks=pytest.mark.slow),  #           #45     # 4.1 Divisible by 32
            pytest.param((64, 160, 96), marks=pytest.mark.slow),  #          #46     # 4.1 Divisible by 32
            pytest.param((11, 17, 41), marks=pytest.mark.run_in_pp),  #      #47     # 4.2 Prime numbers
            pytest.param((13, 89, 3), marks=pytest.mark.slow),  #            #48     # 4.2 Prime numbers
            # 4-dimensional shape, microbatch_size = 1:
            pytest.param(
                (1, 2, 3, 4), marks=pytest.mark.run_in_pp
            ),  #      #49     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param(
                (1, 11, 45, 17), marks=pytest.mark.slow
            ),  #        #50     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param(
                (1, 11, 1, 23), marks=pytest.mark.slow
            ),  #         #51     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param(
                (1, 11, 64, 1), marks=pytest.mark.slow
            ),  #         #52     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param(
                (1, 100, 100, 100), marks=pytest.mark.slow
            ),  #     #53     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param(
                (1, 10, 1000, 100), marks=pytest.mark.slow
            ),  #     #54     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param(
                (1, 1, 10, 1000), marks=pytest.mark.slow
            ),  #       #55     # 4.4 Extreme ratios between height/width
            pytest.param(
                (1, 1, 9920, 1), marks=pytest.mark.slow
            ),  #        #56     # 4.4 Extreme ratios between height/width
            pytest.param(
                (1, 10, 10000, 1), marks=pytest.mark.slow
            ),  #      #57     # 4.4 Extreme ratios between height/width
            pytest.param((1, 32, 32, 64), marks=pytest.mark.run_in_pp),  #   #58     # 4.1 Divisible by 32
            pytest.param((1, 64, 160, 96), marks=pytest.mark.slow),  #       #59     # 4.1 Divisible by 32
            pytest.param((1, 11, 17, 41), marks=pytest.mark.slow),  #        #60     # 4.2 Prime numbers
            pytest.param((1, 13, 89, 3), marks=pytest.mark.slow),  #         #61     # 4.2 Prime numbers
            # 4-dimensional shape, microbatch_size > 1:
            pytest.param(
                (3, 11, 45, 17), marks=pytest.mark.run_in_pp
            ),  #  #62     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param(
                (2, 2, 3, 4), marks=pytest.mark.slow
            ),  #          #63     # 3.1 Full tensor (i.e. full expected shape)
            pytest.param(
                (4, 11, 1, 23), marks=pytest.mark.slow
            ),  #        #64     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param(
                (5, 11, 64, 1), marks=pytest.mark.slow
            ),  #        #65     # 3.2 Tensor reduce on one or more dims to 1
            pytest.param(
                (6, 100, 100, 100), marks=pytest.mark.slow
            ),  #    #66     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param(
                (7, 10, 1000, 100), marks=pytest.mark.slow
            ),  #    #67     # 4.3 Very large (thousands, 10s of thousands)
            pytest.param(
                (8, 1, 10, 1000), marks=pytest.mark.slow
            ),  #      #68     # 4.4 Extreme ratios between height/width
            pytest.param(
                (9, 1, 9920, 1), marks=pytest.mark.slow
            ),  #       #69     # 4.4 Extreme ratios between height/width
            pytest.param(
                (10, 10, 10000, 1), marks=pytest.mark.slow
            ),  #    #70     # 4.4 Extreme ratios between height/width
            pytest.param((11, 32, 32, 64), marks=pytest.mark.slow),  #      #71     # 4.1 Divisible by 32
            pytest.param((12, 64, 160, 96), marks=pytest.mark.slow),  #     #72     # 4.1 Divisible by 32
            pytest.param((13, 11, 17, 41), marks=pytest.mark.run_in_pp),  # #73     # 4.2 Prime numbers
            pytest.param((14, 13, 89, 3), marks=pytest.mark.slow),  #       #74     # 4.2 Prime numbers
        ],
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
            pytest.param(forge.DataFormat.Lf8, id="Lf8"),
            pytest.param(forge.DataFormat.RawUInt8, id="RawUInt8"),
            pytest.param(forge.DataFormat.RawUInt16, id="RawUInt16"),
            pytest.param(forge.DataFormat.RawUInt32, id="RawUInt32"),
            pytest.param(forge.DataFormat.Int8, id="Int8"),
            pytest.param(forge.DataFormat.UInt16, id="UInt16"),
            pytest.param(forge.DataFormat.Int32, id="Int32"),
        ],
        math_fidelities=[
            forge.MathFidelity.LoFi,
            forge.MathFidelity.HiFi2,
            forge.MathFidelity.HiFi3,
            forge.MathFidelity.HiFi4,
        ],
    )

    float = TestCollection(
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
            pytest.param(forge.DataFormat.Lf8, id="Lf8"),
        ],
    )

    int = TestCollection(
        dev_data_formats=[
            pytest.param(forge.DataFormat.RawUInt8, id="RawUInt8"),
            pytest.param(forge.DataFormat.RawUInt16, id="RawUInt16"),
            pytest.param(forge.DataFormat.RawUInt32, id="RawUInt32"),
            pytest.param(forge.DataFormat.Int8, id="Int8"),
            pytest.param(forge.DataFormat.UInt16, id="UInt16"),
            pytest.param(forge.DataFormat.Int32, id="Int32"),
        ],
    )

    quick = TestCollection(
        input_shapes=[]
        + [shape for shape in all.input_shapes if len(shape) in (2,) and shape[0] == 1][:2]
        + [shape for shape in all.input_shapes if len(shape) in (2,) and shape[0] != 1][:2]
        + [shape for shape in all.input_shapes if len(shape) in (3,) and shape[0] == 1][:2]
        + [shape for shape in all.input_shapes if len(shape) in (3,) and shape[0] != 1][:2]
        + [shape for shape in all.input_shapes if len(shape) in (4,) and shape[0] == 1][:2]
        + [shape for shape in all.input_shapes if len(shape) in (4,) and shape[0] != 1][:2],
        dev_data_formats=[
            None,
            forge.DataFormat.Float16_b,
            forge.DataFormat.Int8,
        ],
    )

    specific = TestCollection(
        # input_shapes=[param for param in all.input_shapes if param.marks and pytest.mark.specific in param.marks]
        input_shapes=[
            (1, 45, 17),  # 3.1 Full tensor (i.e. full expected shape)
            (1, 100, 100),  # 4.3 Very large (thousands, 10s of thousands)
            (1, 10000, 1),  # 4.4 Extreme ratios between height/width
            (1, 17, 41),  # 4.2 Prime numbers
            (11, 1, 23),  # 3.2 Tensor reduce on one or more dims to 1
            (1, 11, 1, 23),  # 3.2 Tensor reduce on one or more dims to 1
            (1, 1, 10, 1000),  # 4.4 Extreme ratios between height/width
            (14, 13, 89, 3),  # 4.2 Prime numbers
        ]
    )


class TestCollectionTorch:
    """
    Shared test collection for torch data types.
    """

    __test__ = False  # Avoid collecting TestCollectionTorch as a pytest test

    float = TestCollection(
        dev_data_formats=[
            # torch.float16, # not supported in torch cpu
            torch.float32,
            # torch.float64,
            torch.bfloat16,
        ],
    )

    int = TestCollection(
        dev_data_formats=[
            torch.int8,
            # torch.int16,
            torch.int32,
            torch.int64,
            # torch.uint8,
        ],
    )

    bool = TestCollection(
        dev_data_formats=[
            torch.bool,
        ],
    )

    all = TestCollection(dev_data_formats=float.dev_data_formats + int.dev_data_formats)

    single = TestCollection(
        dev_data_formats=[
            torch.bfloat16,
        ],
    )

    unary_implemented = TestCollection(
        operators=[
            "relu",
            "sqrt",
            "reciprocal",
            "sigmoid",
            "abs",
            "cos",
            "exp",
            "neg",
            "rsqrt",
            "sin",
            "square",
            "pow",
            "clamp",
            "log",
            "log1p",
            "cumsum",
            "isnan",
            "tanh",
        ],
    )

    unary_not_implemented = TestCollection(
        operators=[
            "acos",
            "acosh",
            "angle",
            "asin",
            "asinh",
            "atan",
            "atanh",
            "bitwise_not",
            "ceil",
            "conj_physical",
            "cosh",
            "deg2rad",
            "digamma",
            "erf",
            "erfc",
            "erfinv",
            "exp2",
            "expm1",
            "floor",
            "frac",
            "lgamma",
            "log10",
            "log2",
            "logit",
            "i0",
            "nan_to_num",
            "positive",
            "rad2deg",
            "round",
            "sign",
            "sgn",
            "signbit",
            "sinc",
            "sinh",
            "tan",
            "trunc",
        ],
    )

    unary = TestCollection(
        operators=unary_implemented.operators + unary_not_implemented.operators,
    )

    binary_implemented = TestCollection(
        operators=[
            "add",  #                   #00
            "div",  #                   #01
            "remainder",  #             #12
            "mul",  #                   #03
            "sub",  #                   #05
            # "true_divide",  #         #07     - Alias for div with rounding_mode=None.
            "ge",  #                    #08
            "ne",  #                    #16                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: not_equal      # working with model const
            "gt",  #                    #19                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: greater        # working with model const
            "lt",  #                    #21                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: less           # working with model const
            "maximum",  #               #23                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: maximum        # working with model const
            "minimum",  #               #24                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: minimum        # working with model const
        ],
    )

    binary_not_implemented = TestCollection(
        operators=[
            "atan2",  #                 #00                         - NotImplementedError: The following operators are not implemented: ['aten::atan2']
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
            "fmax",  #                  #13                         - NotImplementedError: The following operators are not implemented: ['aten::fmax']
            "fmin",  #                  #14                         - NotImplementedError: The following operators are not implemented: ['aten::fmin']
            "eq",  #                    #15                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: equal          # working with model const
            "le",  #                    #17                         E       RuntimeError: Unsupported operation for lowering from TTForge to TTIR: less_equal     # working with model const
        ],
    )

    binary = TestCollection(
        operators=binary_implemented.operators + binary_not_implemented.operators,
    )

    bitwise = TestCollection(
        operators=[
            # Unary operators
            "bitwise_not",
            # Binary operators
            "bitwise_and",
            "bitwise_or",
            "bitwise_xor",
            "bitwise_left_shift",
            "bitwise_right_shift",
        ],
    )

    not_implemented = TestCollection(
        operators=unary_not_implemented.operators + binary_not_implemented.operators,
    )
