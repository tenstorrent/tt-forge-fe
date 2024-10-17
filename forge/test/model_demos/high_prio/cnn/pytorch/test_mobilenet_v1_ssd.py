# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import forge

# sys.path = list(set(sys.path + ["third_party/confidential_customer_models/model_2/pytorch/"]))
# from mobilenetv1_ssd.vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd


@pytest.mark.skip(reason="dependent on CCM repo")
def test_mobilenet_v1_ssd_pytorch_1x1(test_device):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    # Load PASCAL VOC dataset class labels
    label_path = "mobilenetv1_ssd/models/voc-model-labels.txt"
    class_names = [name.strip() for name in open(label_path).readlines()]
    number_of_classes = len(class_names)

    # STEP 2: Create Forge module from PyTorch model
    model_path = "mobilenetv1_ssd/models/mobilenet-v1-ssd-mp-0_675.pth"
    net = create_mobilenetv1_ssd(number_of_classes)
    net.load(model_path)
    net.eval()

    input_shape = (1, 3, 300, 300)
    compiled_model = forge.compile(net, sample_inputs=[input_shape])
