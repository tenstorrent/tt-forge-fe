# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import boxes as box_ops


def postprocess_detections(
    head_outputs: dict[str, Tensor], image_anchors: list[Tensor], image_shapes: list[tuple[int, int]]
) -> list[dict[str, Tensor]]:
    bbox_regression = head_outputs["bbox_regression"]
    pred_scores = F.softmax(head_outputs["cls_logits"], dim=-1)

    num_classes = pred_scores.size(-1)
    device = pred_scores.device

    detections: list[dict[str, Tensor]] = []

    for boxes, scores, anchors, image_shape in zip(bbox_regression, pred_scores, image_anchors, image_shapes):
        box_coder = det_utils.BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        boxes = box_coder.decode_single(boxes, anchors)
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        image_boxes = []
        image_scores = []
        image_labels = []
        for label in range(1, num_classes):
            score = scores[:, label]

            keep_idxs = score > 0.01
            score = score[keep_idxs]
            box = boxes[keep_idxs]

            # keep only topk scoring predictions
            num_topk = det_utils._topk_min(score, 400, 0)
            score, idxs = score.topk(num_topk)
            box = box[idxs]

            image_boxes.append(box)
            image_scores.append(score)
            image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64, device=device))

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)

        keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, 0.45)
        keep = keep[:200]

        detections.append(
            {
                "boxes": image_boxes[keep],
                "scores": image_scores[keep],
                "labels": image_labels[keep],
            }
        )
    return detections


class Postprocessor:
    def __init__(self, model):
        super().__init__()
        self.model = model

    def process(self, x, y, images):
        fw_head_outputs = {
            "bbox_regression": x[0],
            "cls_logits": x[1],
        }
        co_head_outputs = {
            "bbox_regression": y[0],
            "cls_logits": y[1],
        }
        anchor_generator = DefaultBoxGenerator(
            [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
            steps=[8, 16, 32, 64, 100, 300],
        )

        image_mean = [0.48235, 0.45882, 0.40784]
        image_std = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]

        transform = GeneralizedRCNNTransform(
            min((300, 300)), max(300, 300), image_mean, image_std, size_divisible=1, fixed_size=(300, 300)
        )
        images, targets = transform(images)
        anchors_fw = anchor_generator(images, [x[2]])
        anchors_co = anchor_generator(images, [y[2]])

        original_image_sizes: list[tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))
        detections_fw = postprocess_detections(fw_head_outputs, anchors_fw, images.image_size)
        detections_fw_op = self.model.transform.postprocess(detections_fw, images.image_size, original_image_sizes)

        detections_co = postprocess_detections(co_head_outputs, anchors_co, images.image_size)
        detections_co_op = self.model.transform.postprocess(detections_co, images.image_size, original_image_sizes)

        return detections_fw_op, detections_co_op
