# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import sys

from loguru import logger

# Add the base directory to sys.path for easier imports
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../"))
logger.info("BASE_DIR in utils={}", BASE_DIR)
MMDET3D_PATH = os.path.join(BASE_DIR, "third_party/mmdetection3d")
logger.info("MMDET3D_PATH in utils={}", MMDET3D_PATH)
sys.path.append(MMDET3D_PATH)

from mmcv import Config
from mmdet3d.datasets import build_dataset

sys.path.append("third_party/models/MapTR/")
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def load_config(variant):
    cfg = Config.fromfile(f"third_party/models/MapTR/projects/configs/maptr/maptr_{variant}.py")
    cfg.data.test.ann_file = "forge/test/models/pytorch/vision/maptr/data/nuscenes/nuscenes_infos_temporal_val.pkl"
    cfg.plugin_dir = "third_party/models/MapTR/projects/mmdet3d_plugin/"
    return cfg


def prepare_model_inputs(cfg):
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    dataset = data_loader.dataset

    for i, data in enumerate(data_loader):

        img_metas = data["img_metas"][0].data[0]
        filename = img_metas[0]["filename"]
        ori_shape = img_metas[0]["ori_shape"]
        img_shape = img_metas[0]["img_shape"]
        lidar2img = img_metas[0]["lidar2img"]
        pad_shape = img_metas[0]["pad_shape"]
        scale_factor = img_metas[0]["scale_factor"]
        flip = img_metas[0]["flip"]
        pcd_horizontal_flip = img_metas[0]["pcd_horizontal_flip"]
        pcd_vertical_flip = img_metas[0]["pcd_vertical_flip"]
        box_mode_3d = img_metas[0]["box_mode_3d"]
        box_type_3d = img_metas[0]["box_type_3d"]
        mean = img_metas[0]["img_norm_cfg"]["mean"]
        std = img_metas[0]["img_norm_cfg"]["std"]
        to_rgb = img_metas[0]["img_norm_cfg"]["to_rgb"]
        sample_idx = img_metas[0]["sample_idx"]
        prev_idx = img_metas[0]["prev_idx"]
        next_idx = img_metas[0]["next_idx"]
        pcd_scale_factor = img_metas[0]["pcd_scale_factor"]
        pts_filename = img_metas[0]["pts_filename"]
        scene_token = img_metas[0]["scene_token"]
        can_bus = img_metas[0]["can_bus"]
        lidar2global = img_metas[0]["lidar2global"]
        camera2ego = img_metas[0]["camera2ego"]
        camera_intrinsics = img_metas[0]["camera_intrinsics"]
        img_aug_matrix = img_metas[0]["img_aug_matrix"]
        lidar2ego = img_metas[0]["lidar2ego"]
        img = data["img"][0].data[0]
        inputs = [img]

        return (
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
        )
