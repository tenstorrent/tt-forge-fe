# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from forge.config import CompileDepth
from forge.verify.backend import verify_module
from test.tvm.recommendation.pytorch.deepctr_torch.layers.interaction import SENETLayer

from forge import (
    PyTorchModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    forge_compile,
)

from test.tvm.utils import evaluate_framework_vs_forge

import torch
import numpy as np
from forge.config import _get_global_compiler_config

def test_fibinet_se(test_kind, test_device):
    # Unsupported HW ops
    _get_global_compiler_config().compile_depth = CompileDepth.FORGE_GRAPH_PRE_PLACER

    filed_size = 32
    batch_size = 1
    embedding_size = 320

    class SENETWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = SENETLayer(filed_size)
        
        def forward(self, x):
            return self.layer(x)

    model = SENETWrapper()
    mod = PyTorchModule("senet_layer", model)
    input_shape = (batch_size,filed_size,embedding_size)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        ),
        uniform_inputs=True,
    )
    # sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    # tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    # tt0.place_module(mod)

    # input_shape = (batch_size,filed_size,embedding_size)
    # inps = [torch.randn(input_shape)]


    # ret = forge_compile(
    #     tt0,
    #     "senet_layer",
    #     *inps,
    #     compiler_cfg=CompilerConfig(
    #         enable_training=training,
    #         enable_recompute=recompute,
    #         compile_depth=compile_depth,
    #     ),
    #     verify_cfg=VerifyConfig(
    #         intermediates=True,
    #     ),
    # )
    # evaluate_framework_vs_forge(model, ret, *inps)
