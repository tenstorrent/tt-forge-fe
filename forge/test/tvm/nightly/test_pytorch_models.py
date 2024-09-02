# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#

import pytest

from urllib.error import HTTPError

from forge import (
    TTDevice,
    BackendType,
    forge_compile,
    VerifyConfig,
    PyTorchModule,
    CompilerConfig,
    CompileDepth,
    optimizers,
)
import forge.compile as COMPILE_INFO

from test.tvm.utils import evaluate_framework_vs_forge
from test.tvm.nightly.get_pytorch_model_with_activations import *


def initialize_device(model_name, model):
    mod = PyTorchModule(model_name, model)

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    return tt0


@pytest.mark.parametrize("mode", ["inference", "training", "recompute"])
@pytest.mark.parametrize("model_name", [x for x in pytorch_model_name_to_forge_model.keys()])
def test_real_networks(mode, model_name):
    if mode == "inference" and model_name in passing_pytorch_model_name_to_forge_model_inference:
        pytest.skip()

    if (mode == "training" or mode == "recompute") and model_name in passing_pytorch_model_name_to_forge_model_training:
        pytest.skip()

    if mode == "inference":
        training = False
        recompute = False
    elif mode == "training":
        training = True
        recompute = False
    elif mode == "recompute":
        training = True
        recompute = True

    # Sometimes the http request to get pretrained models will time out. So, we try a few times if the test fails due to that
    max_http_tries = 3
    http_tries = 0
    while (http_tries < max_http_tries):
        try:
            model_config = pytorch_model_name_to_forge_model[model_name](training, recompute)
            break
        except HTTPError as e:
            http_tries += 1
            if http_tries >= max_http_tries:
                raise e

    waive_gradients = ['key.bias']
    if len(model_config) not in [3, 4]:
        assert False, "Model config must have 3 or 4 attributes, model, inputs, and compiler config"
    
    if len(model_config) == 3:
        model, inputs, compiler_cfg = model_config
    elif len(model_config) == 4:
        model, inputs, compiler_cfg, waive_extra_gradients = model_config
        waive_gradients.extend(waive_extra_gradients)

    tt_device = initialize_device(model_name, model)

    try:
        ret = forge_compile(
            tt_device,
            model_name,
            inputs[0],
            compiler_cfg=compiler_cfg,
            verify_cfg=VerifyConfig(intermediates=True, verify_last=False, waive_gradient_errors=waive_gradients),
        )

        evaluate_framework_vs_forge(model, ret, *inputs)
    except Exception as e:
        pytest.fail(
            msg=f"Last completed compile stage: {COMPILE_INFO.LAST_SUCCESSFUL_STAGE}. Error: {e}",
            pytrace=False,
        )
