# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from easydict import EasyDict as edict

import forge

from test.models.pytorch.vision.complex_yolov4.utils.model_utils import create_model


@pytest.mark.parametrize("variant", ["complex_yolov4_tiny", "complex_yolov4"])
# @pytest.mark.parametrize("variant", ["complex_yolov4_tiny"])
def test_compelx_yolov4(variant):

    configs = edict(
        {
            "arch": "darknet",
            "cfgfile": f"forge/test/models/pytorch/vision/complex_yolov4/utils/{variant}.cfg",
        }
    )

    model = create_model(configs)
    model.eval()

    sample_input = torch.randn((1, 3, 608, 608))

    # os.environ["FORGE_EXTRACT_UNIQUE_OP_CONFIG_AT"] = "ALL"
    # os.environ["FORGE_PRINT_UNIQUE_OP_CONFIG"] = "1"

    # with torch.no_grad():
    # output = model(sample_input)

    # print("sample_input",sample_input)
    # print("model",model)
    # print("output.shape",output.shape)
    # print("output",output)

    # print("========================================")

    compiled_model = forge.compile(model, sample_inputs=[sample_input], module_name="pt_yolov4_complex")
