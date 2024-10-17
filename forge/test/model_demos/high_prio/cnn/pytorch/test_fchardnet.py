import forge
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
# from forge.verify.backend import verify_module
# from forge.verify.config import TestKind
# from forge._C.backend_api import BackendDevice
# from forge import VerifyConfig
import sys

sys.path.append("tt-forge-fe/forge/test/model_demos/high_prio/cnn/pytorch/model2/pytorch/fchardnet")
from model_fchardnet import get_model, fuse_bn_recursively


def test_fchardnet(test_device):
    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = forge.DataFormat.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"

    # Load and pre-process image
    image_path = "tt-forge-fe/forge/test/model_demos/high_prio/cnn/pytorch/model2/pytorch/pidnet/image/road_scenes.png"
    img = Image.open(image_path)
    img = np.array(img.resize((320, 320)), dtype=np.uint8)
    img = img[:, :, ::-1]
    mean = np.array([0.406, 0.456, 0.485]) * 255
    std = np.array([0.225, 0.224, 0.229]) * 255
    img = (img.astype(np.float64) - mean) / std
    img = torch.tensor(img).float().permute(2, 0, 1)
    input_image = img.unsqueeze(0)

    # Load model
    device = torch.device("cpu")
    arch = {"arch": "hardnet"}
    model = get_model(arch, 19).to(device)
    model = fuse_bn_recursively(model)
    model.eval()
    # tt_model = forge.PyTorchModule("fchardnet", model)
    compiled_model = forge.compile(model, sample_inputs=[input_image])

    # Verify
    # verify_module(
    #     tt_model,
    #     input_shapes=(input_image.shape),
    #     inputs=[(input_image)],
    #     verify_cfg=VerifyConfig(
    #         arch=test_device.arch,
    #         devtype=test_device.devtype,
    #         devmode=test_device.devmode,
    #         test_kind=TestKind.INFERENCE,
    #     ),
    # )
