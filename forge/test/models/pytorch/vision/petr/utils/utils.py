# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from mmcv import Config
from mmdet3d.datasets import build_dataloader, build_dataset

from test.models.pytorch.vision.petr.mmdet3d.core.bbox.transforms import bbox3d2result


def load_config(variant):
    cfg = Config.fromfile(f"forge/test/models/pytorch/vision/petr/utils/petr_{variant}.py")
    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = "forge/test/models/pytorch/vision/petr/data/nuscenes/nuscenes_infos_val.pkl"
    return cfg


def prepare_model_inputs(cfg):

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
    dataset = data_loader.dataset

    for i, data in enumerate(data_loader):

        img_metas = data["img_metas"][0].data[0]
        filename = img_metas[0]["filename"]
        ori_shape = img_metas[0]["ori_shape"]
        img_shape = img_metas[0]["img_shape"]
        pad_shape = img_metas[0]["pad_shape"]
        scale_factor = img_metas[0]["scale_factor"]
        flip = img_metas[0]["flip"]
        pcd_horizontal_flip = img_metas[0]["pcd_horizontal_flip"]
        pcd_vertical_flip = img_metas[0]["pcd_vertical_flip"]
        box_mode_3d = img_metas[0]["box_mode_3d"]
        box_type_3d = img_metas[0]["box_type_3d"]
        mean = torch.from_numpy(img_metas[0]["img_norm_cfg"]["mean"])
        std = torch.from_numpy(img_metas[0]["img_norm_cfg"]["std"])
        to_rgb = img_metas[0]["img_norm_cfg"]["to_rgb"]
        sample_idx = img_metas[0]["sample_idx"]
        pcd_scale_factor = img_metas[0]["pcd_scale_factor"]
        pts_filename = img_metas[0]["pts_filename"]
        img = data["img"][0].data[0]
        lidar2img_list = img_metas[0]["lidar2img"]

        lidar2img_tensors_list = []

        for idx, lidar2img_array in enumerate(lidar2img_list):
            lidar2img_tensor = torch.from_numpy(lidar2img_array)
            lidar2img_tensors_list.append(lidar2img_tensor)

        batch_size = 1
        num_cams = 6
        input_img_h, input_img_w, _ = pad_shape[0]
        x = torch.rand(batch_size, num_cams, input_img_h, input_img_w)
        masks = x.new_ones((batch_size, num_cams, input_img_h, input_img_w))

        inputs = [
            lidar2img_tensors_list[0].unsqueeze(0),
            lidar2img_tensors_list[1].unsqueeze(0),
            lidar2img_tensors_list[2].unsqueeze(0),
            lidar2img_tensors_list[3].unsqueeze(0),
            lidar2img_tensors_list[4].unsqueeze(0),
            lidar2img_tensors_list[5].unsqueeze(0),
            img.unsqueeze(0),
            mean.unsqueeze(0),
            std.unsqueeze(0),
            masks.unsqueeze(0),
        ]

        for i, tensor in enumerate(inputs):
            if tensor.dtype == torch.float64:
                inputs[i] = tensor.to(torch.float32)

        return (
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
        )


def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp()
    l = l.exp()
    h = h.exp()
    if normalized_bboxes.size(-1) > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes


def decode_single(cls_scores, bbox_preds):

    # post processing
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    voxel_size = [0.2, 0.2, 8]
    post_center_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
    max_num = 300
    score_threshold = None
    num_classes = 10

    cls_scores = cls_scores.sigmoid()
    scores, indexs = cls_scores.view(-1).topk(max_num)
    labels = indexs % num_classes
    bbox_index = indexs // num_classes

    bbox_preds = bbox_preds[bbox_index]

    final_box_preds = denormalize_bbox(bbox_preds, pc_range)
    final_scores = scores
    final_preds = labels

    if score_threshold is not None:
        thresh_mask = final_scores > self.score_threshold
    if post_center_range is not None:
        post_center_range = torch.tensor(post_center_range, device=scores.device)

        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(1)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(1)

        if score_threshold:
            mask &= thresh_mask

        boxes3d = final_box_preds[mask]
        scores = final_scores[mask]
        labels = final_preds[mask]
        predictions_dict = {"bboxes": boxes3d, "scores": scores, "labels": labels}

        return predictions_dict


def post_process(img_metas, all_class_scores, all_bbox_predictions):

    all_cls_scores = all_class_scores[-1]
    all_bbox_preds = all_bbox_predictions[-1]

    batch_size = all_cls_scores.size()[0]
    predictions_list = []
    for i in range(batch_size):
        predictions_list.append(decode_single(all_cls_scores[i], all_bbox_preds[i]))

    preds_dicts = predictions_list
    num_samples = len(preds_dicts)
    ret_list = []
    for i in range(num_samples):
        preds = preds_dicts[i]
        bboxes = preds["bboxes"]
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        bboxes = img_metas[i]["box_type_3d"](bboxes, bboxes.size(-1))
        scores = preds["scores"]
        labels = preds["labels"]
        ret_list.append([bboxes, scores, labels])

    bbox_list = ret_list
    bbox_results = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]

    bbox_list = [dict() for i in range(1)]
    bbox_pts = bbox_results
    for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
        result_dict["pts_bbox"] = pts_bbox

    boxes_3d_tensor = bbox_list[0]["pts_bbox"]["boxes_3d"].tensor
    scores_3d_tensor = bbox_list[0]["pts_bbox"]["scores_3d"]
    labels_3d_tensor = bbox_list[0]["pts_bbox"]["labels_3d"]

    output = (boxes_3d_tensor, scores_3d_tensor, labels_3d_tensor)

    return output
