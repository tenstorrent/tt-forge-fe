# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import sys

from loguru import logger

# Add the base directory to sys.path for easier imports
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../"))
logger.info("BASE_DIR={}", BASE_DIR)
MMDET3D_PATH = os.path.join(BASE_DIR, "third_party/mmdetection3d")
logger.info("MMDET3D_PATH={}", MMDET3D_PATH)
sys.path.append(MMDET3D_PATH)

import pytest
import torch
from mmcv.parallel import MMDataParallel
from mmdet3d.models import build_model

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.maptr.utils.utils import (
    load_config,
    prepare_model_inputs,
)
from test.models.utils import Framework, Source, Task, build_module_name

variants = ["tiny_r50_24e_bevformer", "tiny_r50_24e_bevformer_t4"]


class maptr_wrapper(torch.nn.Module):
    def __init__(
        self,
        model,
        filename,
        ori_shape,
        img_shape,
        lidar2img,
        pad_shape,
        scale_factor,
        flip,
        pcd_horizontal_flip,
        pcd_vertical_flip,
        box_mode_3d,
        box_type_3d,
        mean,
        std,
        to_rgb,
        sample_idx,
        prev_idx,
        next_idx,
        pcd_scale_factor,
        pts_filename,
        scene_token,
        can_bus,
        lidar2global,
        camera2ego,
        camera_intrinsics,
        img_aug_matrix,
        lidar2ego,
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
        self.prev_idx = prev_idx
        self.next_idx = next_idx
        self.scene_token = scene_token
        self.pcd_scale_factor = pcd_scale_factor
        self.pts_filename = pts_filename
        self.can_bus = can_bus
        self.lidar2img = lidar2img
        self.lidar2global = lidar2global
        self.camera2ego = camera2ego
        self.camera_intrinsics = camera_intrinsics
        self.img_aug_matrix = img_aug_matrix
        self.lidar2ego = lidar2ego
        self.mean = mean
        self.std = std

    def forward(self, img):

        data = {
            "img_metas": [
                [
                    {
                        "filename": self.filename,
                        "ori_shape": self.ori_shape,
                        "img_shape": self.img_shape,
                        "lidar2img": self.lidar2img,
                        "pad_shape": self.pad_shape,
                        "scale_factor": self.scale_factor,
                        "flip": self.flip,
                        "pcd_horizontal_flip": self.pcd_horizontal_flip,
                        "pcd_vertical_flip": self.pcd_vertical_flip,
                        "box_mode_3d": self.box_mode_3d,
                        "box_type_3d": self.box_type_3d,
                        "img_norm_cfg": {"mean": self.mean, "std": self.std, "to_rgb": self.to_rgb},
                        "sample_idx": self.sample_idx,
                        "prev_idx": self.prev_idx,
                        "next_idx": self.next_idx,
                        "pcd_scale_factor": self.pcd_scale_factor,
                        "pts_filename": self.pts_filename,
                        "scene_token": self.scene_token,
                        "can_bus": self.can_bus,
                        "lidar2global": self.lidar2global,
                        "camera2ego": self.camera2ego,
                        "camera_intrinsics": self.camera_intrinsics,
                        "img_aug_matrix": self.img_aug_matrix,
                        "lidar2ego": self.lidar2ego,
                    }
                ]
            ],
            "img": [img],
        }

        outputs = self.model(return_loss=False, rescale=True, **data)

        boxes_3d = outputs[0]["pts_bbox"]["boxes_3d"]
        scores_3d = outputs[0]["pts_bbox"]["scores_3d"]
        labels_3d = outputs[0]["pts_bbox"]["labels_3d"]
        pts_3d = outputs[0]["pts_bbox"]["pts_3d"]
        return (boxes_3d, scores_3d, labels_3d, pts_3d)


@pytest.mark.parametrize("variant", variants)
def test_maptr(record_forge_property, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="maptr",
        variant=variant,
        source=Source.GITHUB,
        task=Task.OBJECT_DETECTION,
    )

    # Record Forge Property
    record_forge_property("group", "priority")
    record_forge_property("tags.model_name", module_name)

    # Load config
    cfg = load_config(variant)

    # Prepare input
    (
        filename,
        ori_shape,
        img_shape,
        lidar2img,
        pad_shape,
        scale_factor,
        flip,
        pcd_horizontal_flip,
        pcd_vertical_flip,
        box_mode_3d,
        box_type_3d,
        mean,
        std,
        to_rgb,
        sample_idx,
        prev_idx,
        next_idx,
        pcd_scale_factor,
        pts_filename,
        scene_token,
        can_bus,
        lidar2global,
        camera2ego,
        camera_intrinsics,
        img_aug_matrix,
        lidar2ego,
        inputs,
    ) = prepare_model_inputs(cfg)

    # Load Model
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    framework_model = maptr_wrapper(
        model,
        filename,
        ori_shape,
        img_shape,
        lidar2img,
        pad_shape,
        scale_factor,
        flip,
        pcd_horizontal_flip,
        pcd_vertical_flip,
        box_mode_3d,
        box_type_3d,
        mean,
        std,
        to_rgb,
        sample_idx,
        prev_idx,
        next_idx,
        pcd_scale_factor,
        pts_filename,
        scene_token,
        can_bus,
        lidar2global,
        camera2ego,
        camera_intrinsics,
        img_aug_matrix,
        lidar2ego,
    )
    framework_model.eval()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
