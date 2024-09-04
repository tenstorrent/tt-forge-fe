# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of matmul operators
#
# In this test we use pytorch tensors and operators to verify forge operators
#

import os
import pytest
import numpy as np

import forge
import forge.op
from forge import TTDevice, BackendType, forge_compile, VerifyConfig, CompilerConfig

from .models import generic

MODELS_PATH = "./forge/test/operators/matmul/models/"
MODELS_GENERIC_PATH = MODELS_PATH + "generic/" 

# @pytest.mark.xfail(
#     reason="tenstorrent/forge#5"
# )
def test_matmul_generic(
    mm_train,
    mm_recompute,
    mm_model,
    mm_shape
):

    print("\n")
    print(f"mm_train --> {mm_train}")
    print(f"mm_recompute --> {mm_recompute}")
    print(f"mm_model --> {mm_model}")
    print(f"mm_shape --> {mm_shape}")
    print("\n")

    if mm_train and len(mm_shape) >= 3 and mm_shape[-3] > 1:
        pytest.skip("Matmul with gradient accumulate must have t=1")

    if not mm_train and mm_recompute:
        pytest.skip("Inference and recompute is the same as just inference.")
    
    assert type(mm_train) in [bool, str], "Type of training parameter must be boolean or string"
    if type(mm_train) == str:
        training = True if mm_train == 'True' else False
    else:
        training = mm_train
    assert type(mm_recompute) in [bool, str], "Type of recompute parameter must be boolean or string"
    if type(mm_recompute) == str:
        recompute = True if mm_recompute == 'True' else False
    else:
        recompute = mm_recompute
    model = mm_model
    shape = eval(mm_shape) if type(mm_shape) == str else mm_shape

    print("\n")
    print(f"Training --> {training}")
    print(f"Recompute --> {recompute}")
    print(f"Model --> {model}")
    print(f"Shape --> {shape}")
    print("\n")

    architecture = f'generic.{model}.ForgeMatmulTest(shape={shape})'
    model = eval(architecture)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(model)
    forge_compile(
        tt0, 
        model.testname, 
        *model.inputs, 
        compiler_cfg=CompilerConfig(
                        enable_training=training,
                        enable_recompute=recompute
                     ), 
        verify_cfg=VerifyConfig()
    )
