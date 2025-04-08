import cv2
import numpy as np
import pyclipper
import paddle
from shapely.geometry import Polygon

min_size = 4
thresh = 0.2
box_thresh = 0.6
max_candidates = 1000
unclip_ratio = 1.7


def get_mini_boxes(contour):
    """
    Get the minimum area bounding box for a contour with fixed orientation.
    """
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])

def bitmap_from_probmap(preds):
    prob_map = preds.numpy()[0, 0]  # (H, W)
    bitmap = (prob_map > 0.3).astype(np.uint8)
    bitmap = np.ascontiguousarray(bitmap)
    return bitmap

def boxes_from_bitmap(pred, bitmap, dest_width, dest_height):
    """
    _bitmap: single map with shape (H, W),
        whose values are binarized as {0, 1}
    """

    assert len(bitmap.shape) == 2
    height, width = bitmap.shape
    contours, _ = cv2.findContours(
        (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    num_contours = min(len(contours), max_candidates)
    boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
    scores = np.zeros((num_contours,), dtype=np.float32)

    for index in range(num_contours):
        contour = contours[index].squeeze(1)
        points, sside = get_mini_boxes(contour)
        if sside < min_size:
            continue
        points = np.array(points)
        score = box_score_fast(bitmap, points)
        if box_thresh > score:
            continue

        box = unclip(points, unclip_ratio=unclip_ratio).reshape(-1, 1, 2)
        box, sside = get_mini_boxes(box)
        if sside < min_size + 2:
            continue
        box = np.array(box)
        if not isinstance(dest_width, int):
            dest_width = dest_width.item()
            dest_height = dest_height.item()

        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(
            np.round(box[:, 1] / height * dest_height), 0, dest_height
        )
        boxes[index, :, :] = box.astype(np.int16)
        scores[index] = score

    # Filter boxes with scores > 0
    valid_indices = scores > 0
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    return boxes, scores

def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    img_copy = image.copy()

    for box in boxes:
        box = np.int32(box)
        pts = box.reshape((-1, 1, 2))
        cv2.polylines(img_copy, [pts], isClosed=True, color=color, thickness=thickness)

    return img_copy

def cut_boxes(image, boxes):
    small_images = []
    eps = 10
    for box in boxes:
        box = np.int32(box)
        x_min = max(0, np.min(box[:, 0]))
        x_max = min(image.shape[1], np.max(box[:, 0]) + eps)
        y_min = max(0, np.min(box[:, 1]) - eps)
        y_max = min(image.shape[0], np.max(box[:, 1]) + eps)
        x_min = max(0, np.min(box[:, 0]) - eps)
        cropped_image = image[y_min:y_max, x_min:x_max]
        small_images.append(cropped_image)
    return small_images
    
