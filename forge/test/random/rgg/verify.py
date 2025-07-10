# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Verify model for RGG

from typing import List

from forge import ForgeModule
from forge import Module
from forge import TensorShape

from forge.verify.config import VerifyConfig
from forge.config import CompilerConfig

# from forge.verify.value_checkers import AllCloseValueChecker
from forge.verify.value_checkers import AutomaticValueChecker

from test.operators.utils import VerifyUtils
from test.operators.utils import ValueRanges

# from test.operators.utils import FrameworkDataFormat


# TODO FrameworkModel


# Common verification method for RGG
def verify_module(
    model: Module,
    input_shapes: List[TensorShape],
    # dev_data_format: FrameworkDataFormat,
    random_seed: int,
):
    # dev_data_format: FrameworkDataFormat
    # dev_data_format = torch.bfloat16
    # dev_data_format = torch.float32
    # dev_data_format = None
    # dev_data_format = forge.DataFormat.Float32

    compiler_cfg = CompilerConfig()
    # Reset default data format to None set by conftest.py
    # compiler_cfg.default_df_override = None

    VerifyUtils.verify(
        model=model,
        test_device=None,
        input_shapes=input_shapes,
        compiler_cfg=compiler_cfg,
        input_source_flag=None,
        # dev_data_format=dev_data_format,
        math_fidelity=None,
        # value_range=ValueRanges.SMALL,
        # value_range=ValueRanges.SMALL_POSITIVE,
        value_range=ValueRanges.LARGE,
        random_seed=random_seed,
        # verify_config=VerifyConfig(value_checker=AllCloseValueChecker(rtol=1e-02, atol=1e-02)),
        verify_config=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.99)),
    )
