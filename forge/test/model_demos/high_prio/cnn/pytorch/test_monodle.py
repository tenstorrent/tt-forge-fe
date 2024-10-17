import forge
# from forge.verify.backend import verify_module
# from forge import VerifyConfig
# from forge.verify import TestKind
import os
import requests
import torchvision.transforms as transforms
from PIL import Image
from test.model_demos.models.monodle import CenterNet3D


def test_monodle_pytorch(test_device):
    # PyBuda configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.default_df_override = forge._C.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"

    pcc = 0.99
    # if test_device.arch == forge.BackendDevice.Wormhole_B0:
    #     os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"
    #     os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{88*1024}"
    #     pcc = 0.98
    # elif test_device.arch == forge.BackendDevice.Grayskull:
    #     os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"
    #     pcc = 0.93
    # elif test_device.arch == forge.BackendDevice.Blackhole:
    #     pcc = 0.97

    model_name = "monodle_pytorch"

    # Load data sample
    url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = transform(image).unsqueeze(0)

    pytorch_model = CenterNet3D(backbone="dla34")
    pytorch_model.eval()

    # tt_model = forge.PyTorchModule(model_name, pytorch_model)
    compiled_model = forge.compile(pytorch_model, sample_inputs=[img_tensor])

    # verify_module(
    #     tt_model,
    #     input_shapes=[img_tensor.shape],
    #     inputs=[(img_tensor,)],
    #     verify_cfg=VerifyConfig(
    #         arch=test_device.arch,
    #         devtype=test_device.devtype,
    #         devmode=test_device.devmode,
    #         test_kind=TestKind.INFERENCE,
    #         pcc=pcc,
    #     ),
    # )
