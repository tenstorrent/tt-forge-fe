# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
import pytest
import forge
import torch
import os
import sys

# sys.path = list(set(sys.path + ["third_party/confidential_customer_models/model_2/pytorch/"]))
# from mobilenetv1_ssd.vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_mobilenet_v1_ssd_pytorch_1x1(test_device):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # Load PASCAL VOC dataset class labels
    label_path = "third_party/confidential_customer_models/model_2/pytorch/mobilenetv1_ssd/models/voc-model-labels.txt"
    class_names = [name.strip() for name in open(label_path).readlines()]
    number_of_classes = len(class_names)

    # STEP 2: Create Forge module from PyTorch model
    model_path = (
        "third_party/confidential_customer_models/model_2/pytorch/mobilenetv1_ssd/models/mobilenet-v1-ssd-mp-0_675.pth"
    )
    net = create_mobilenetv1_ssd(number_of_classes)
    net.load(model_path)
    net.eval()

    input_shape = (1, 3, 300, 300)
    inputs = [torch.rand(input_shape)]
    compiled_model = forge.compile(net, sample_inputs=inputs, module_name="pt_mobilenet_v1_ssd")
