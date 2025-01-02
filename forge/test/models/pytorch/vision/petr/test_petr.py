# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import sys

import torch

sys.path.append("forge/test/models/pytorch/vision/petr")
import pytest
from mmcv.parallel import MMDataParallel
from mmdet3d.models.builder import build_model

import forge
from forge.verify.verify import verify

# Import necessary classes for model registration, ensuring availability even if not used directly
from utils import model_registry
from utils.utils import load_config, prepare_model_inputs

from test.models.utils import Framework, Source, Task, build_module_name


class petr_wrapper(torch.nn.Module):
    def __init__(
        self,
        model,
        filename,
        ori_shape,
        img_shape,
        pad_shape,
        scale_factor,
        flip,
        pcd_horizontal_flip,
        pcd_vertical_flip,
        box_mode_3d,
        box_type_3d,
        to_rgb,
        sample_idx,
        pcd_scale_factor,
        pts_filename,
    ):
        super().__init__()
        self.model = model

        self.filename = filename
        self.ori_shape = ori_shape
        self.img_shape = img_shape
        self.pad_shape = pad_shape
        self.scale_factor = scale_factor
        self.flip = flip
        self.pcd_horizontal_flip = pcd_horizontal_flip
        self.pcd_vertical_flip = pcd_vertical_flip
        self.box_mode_3d = box_mode_3d
        self.box_type_3d = box_type_3d
        self.to_rgb = to_rgb
        self.sample_idx = sample_idx
        self.pcd_scale_factor = pcd_scale_factor
        self.pts_filename = pts_filename

    def forward(self, l0, l1, l2, l3, l4, l5, img, mean, std, masks):

        l0 = l0.squeeze(0)
        l1 = l1.squeeze(0)
        l2 = l2.squeeze(0)
        l3 = l3.squeeze(0)
        l4 = l4.squeeze(0)
        l5 = l5.squeeze(0)
        img = img.squeeze(0)
        mean = mean.squeeze(0)
        std = std.squeeze(0)
        masks = masks.squeeze(0)

        data = {
            "img_metas": [
                [
                    {
                        "filename": self.filename,
                        "ori_shape": self.ori_shape,
                        "img_shape": self.img_shape,
                        "lidar2img": [l0, l1, l2, l3, l4, l5],
                        "pad_shape": self.pad_shape,
                        "scale_factor": self.scale_factor,
                        "flip": self.flip,
                        "pcd_horizontal_flip": self.pcd_horizontal_flip,
                        "pcd_vertical_flip": self.pcd_vertical_flip,
                        "box_mode_3d": self.box_mode_3d,
                        "box_type_3d": self.box_type_3d,
                        "img_norm_cfg": {"mean": mean, "std": std, "to_rgb": self.to_rgb},
                        "sample_idx": self.sample_idx,
                        "pcd_scale_factor": self.pcd_scale_factor,
                        "pts_filename": self.pts_filename,
                        "masks": masks,
                    }
                ]
            ],
            "img": [img],
        }

        output = self.model(**data)
        return (output["all_cls_scores"], output["all_bbox_preds"])


variants = ["vovnet_gridmask_p4_800x320", "vovnet_gridmask_p4_1600x640"]


@pytest.mark.parametrize("variant", variants)
def test_petr(record_forge_property, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="petr", source=Source.GITHUB, task=Task.OBJECT_DETECTION, variant=variant
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    _ = model_registry  # Prevents removal by linters/formatters

    # Load config
    cfg = load_config(variant)

    # Prepare input
    (
        filename,
        ori_shape,
        img_shape,
        pad_shape,
        scale_factor,
        flip,
        pcd_horizontal_flip,
        pcd_vertical_flip,
        box_mode_3d,
        box_type_3d,
        to_rgb,
        sample_idx,
        pcd_scale_factor,
        pts_filename,
        inputs,
    ) = prepare_model_inputs(cfg)

    # Load Model
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"), train_cfg=cfg.get("train_cfg"))
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    framework_model = petr_wrapper(
        model,
        filename,
        ori_shape,
        img_shape,
        pad_shape,
        scale_factor,
        flip,
        pcd_horizontal_flip,
        pcd_vertical_flip,
        box_mode_3d,
        box_type_3d,
        to_rgb,
        sample_idx,
        pcd_scale_factor,
        pts_filename,
    )
    framework_model.eval()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
