# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import forge
import forge.op
from forge import ForgeModule

from loguru import logger
import torch

from forge import Tensor, compile
from forge.verify.verify import verify
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.config import VerifyConfig
from forge.forge_property_utils import (
    record_forge_op_name,
    record_op_model_names,
    record_forge_op_args,
    record_single_op_operands_info,
)
import pytest


class Resize2D0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[120, 120], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[30, 30], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[60, 60], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[128, 128], method="linear", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[112, 112], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[28, 28], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[56, 56], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[27, 40], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[54, 80], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[107, 160], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[40, 40], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[80, 80], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Resize2D0,
        [((1, 24, 15, 15), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "sizes": "[120, 120]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D1,
        [((1, 96, 15, 15), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "sizes": "[30, 30]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D0,
        [((1, 24, 30, 30), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "sizes": "[120, 120]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D2,
        [((1, 96, 30, 30), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "sizes": "[60, 60]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D0,
        [((1, 24, 60, 60), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "sizes": "[120, 120]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D0,
        [((1, 96, 60, 60), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "sizes": "[120, 120]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D3,
        [((1, 768, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"sizes": "[128, 128]", "method": '"linear"', "align_corners": "False", "channel_last": "0"},
        },
    ),
    (
        Resize2D3,
        [((1, 768, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"sizes": "[128, 128]", "method": '"linear"', "align_corners": "False", "channel_last": "0"},
        },
    ),
    (
        Resize2D3,
        [((1, 768, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"sizes": "[128, 128]", "method": '"linear"', "align_corners": "False", "channel_last": "0"},
        },
    ),
    (
        Resize2D3,
        [((1, 768, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"sizes": "[128, 128]", "method": '"linear"', "align_corners": "False", "channel_last": "0"},
        },
    ),
    (
        Resize2D4,
        [((1, 24, 14, 14), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "sizes": "[112, 112]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D5,
        [((1, 96, 14, 14), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D4,
        [((1, 24, 28, 28), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "sizes": "[112, 112]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D6,
        [((1, 96, 28, 28), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D4,
        [((1, 24, 56, 56), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "sizes": "[112, 112]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D4,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "sizes": "[112, 112]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((100, 128, 14, 20), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {
                "sizes": "[27, 40]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((100, 64, 27, 40), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {
                "sizes": "[54, 80]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D9,
        [((100, 32, 54, 80), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {
                "sizes": "[107, 160]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D10,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "sizes": "[40, 40]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D11,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "sizes": "[80, 80]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D3,
        [((1, 256, 16, 16), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"sizes": "[128, 128]", "method": '"linear"', "align_corners": "False", "channel_last": "0"},
        },
    ),
    (
        Resize2D3,
        [((1, 256, 32, 32), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"sizes": "[128, 128]", "method": '"linear"', "align_corners": "False", "channel_last": "0"},
        },
    ),
    (
        Resize2D3,
        [((1, 256, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"sizes": "[128, 128]", "method": '"linear"', "align_corners": "False", "channel_last": "0"},
        },
    ),
    (
        Resize2D3,
        [((1, 256, 128, 128), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"sizes": "[128, 128]", "method": '"linear"', "align_corners": "False", "channel_last": "0"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Resize2d")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name == "model_names":
            record_op_model_names(metadata_value)
        elif metadata_name == "args":
            record_forge_op_args(metadata_value)
        else:
            logger.warning(
                "No utility function available in forge property handler to record %s property", metadata_name
            )

    max_int = 1000
    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype, max_int=max_int)
        for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)
    framework_model.process_framework_parameters()

    for name, parameter in framework_model._parameters.items():
        parameter_tensor = Tensor.create_torch_tensor(
            shape=parameter.shape.get_pytorch_shape(), dtype=parameter.pt_data_format, max_int=max_int
        )
        framework_model.set_parameter(name, parameter_tensor)

    for name, constant in framework_model._constants.items():
        constant_tensor = Tensor.create_torch_tensor(
            shape=constant.shape.get_pytorch_shape(), dtype=constant.pt_data_format, max_int=max_int
        )
        framework_model.set_constant(name, constant_tensor)

    record_single_op_operands_info(framework_model, inputs)

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
