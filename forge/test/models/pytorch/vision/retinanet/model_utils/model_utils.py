# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from typing import Dict, List

import torch
from torch import Tensor
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import boxes as box_ops


def _default_anchorgen():
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    return anchor_generator


def split_outputs_by_level(feature, head_outputs, anchors):
    # Calculate number of anchors per level based on feature map sizes
    num_anchors_per_level = [x.size(2) * x.size(3) for x in feature]
    HW = sum(num_anchors_per_level)
    # Total anchors per image
    HWA = head_outputs["cls_logits"].size(1)
    A = HWA // HW
    # Update anchor count with A (number of anchors per spatial location)
    num_anchors_per_level = [hw * A for hw in num_anchors_per_level]
    # Split outputs per level
    split_head_outputs: Dict[str, List[Tensor]] = {
        k: list(v.split(num_anchors_per_level, dim=1)) for k, v in head_outputs.items()
    }
    # Split anchors per level
    split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]
    return split_head_outputs, split_anchors


def postprocess_detections(head_outputs, anchors, image_shapes):
    # type: (dict[str, list[Tensor]], list[list[Tensor]], list[tuple[int, int]]) -> list[dict[str, Tensor]]
    class_logits = head_outputs["cls_logits"]
    box_regression = head_outputs["bbox_regression"]

    num_images = len(image_shapes)

    detections: list[dict[str, Tensor]] = []

    for index in range(num_images):
        box_regression_per_image = [br[index] for br in box_regression]
        logits_per_image = [cl[index] for cl in class_logits]
        anchors_per_image, image_shape = anchors[index], image_shapes[index]

        image_boxes = []
        image_scores = []
        image_labels = []

        for box_regression_per_level, logits_per_level, anchors_per_level in zip(
            box_regression_per_image, logits_per_image, anchors_per_image
        ):
            box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
            num_classes = logits_per_level.shape[-1]

            # remove low scoring boxes
            scores_per_level = torch.sigmoid(logits_per_level).flatten()
            keep_idxs = scores_per_level > 0.05
            scores_per_level = scores_per_level[keep_idxs]
            topk_idxs = torch.where(keep_idxs)[0]

            # keep only topk scoring predictions
            num_topk = det_utils._topk_min(topk_idxs, 1000, 0)
            scores_per_level, idxs = scores_per_level.topk(num_topk)
            topk_idxs = topk_idxs[idxs]

            anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
            labels_per_level = topk_idxs % num_classes

            boxes_per_level = box_coder.decode_single(
                box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
            )
            boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

            image_boxes.append(boxes_per_level)
            image_scores.append(scores_per_level)
            image_labels.append(labels_per_level)

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)

        # non-maximum suppression
        keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, 0.5)
        keep = keep[:300]

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
        anchor_generator = _default_anchorgen()

        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]

        transform = GeneralizedRCNNTransform(800, 1333, image_mean, image_std)

        images, targets = transform(images)
        anchors_fw = anchor_generator(images, [x[2]])
        anchors_co = anchor_generator(images, [y[2]])

        original_image_sizes: list[tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        split_head_outputs_fw, split_anchors_fw = split_outputs_by_level(
            feature=x[2], head_outputs=fw_head_outputs, anchors=anchors_fw
        )

        split_head_outputs_co, split_anchors_co = split_outputs_by_level(
            feature=y[2], head_outputs=co_head_outputs, anchors=anchors_co
        )

        detections_fw = postprocess_detections(split_head_outputs_fw, split_anchors_fw, images.image_size)
        detections_fw_op = self.model.transform.postprocess(detections_fw, images.image_size, original_image_sizes)

        detections_co = postprocess_detections(split_head_outputs_co, split_anchors_co, images.image_size)
        detections_co_op = self.model.transform.postprocess(detections_co, images.image_size, original_image_sizes)

        return detections_fw_op, detections_co_op
