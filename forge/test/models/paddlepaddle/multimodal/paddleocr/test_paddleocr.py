# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import paddle
import pytest
import cv2

import forge
from forge.verify.verify import verify

from test.utils import fetch_paddle_model
from test.models.paddlepaddle.multimodal.paddleocr.det_post_processing import bitmap_from_probmap, draw_boxes, cut_boxes, polygons_from_bitmap, process_and_pad_images
from enum import Enum
import argparse

model_urls = {
    "v4": {
        "det": {
            "ch": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar"},
            "en": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_det_infer.tar"},
            "ml": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar"},
        },
        "rec": {
            "ch": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar",
                   "dict_path": "forge/test/models/paddlepaddle/multimodal/paddleocr/cached_models/ppocr_keys_v1.txt"},
            "en": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
                     "dict_path": "forge/test/models/paddlepaddle/multimodal/paddleocr/cached_models/en_dict.txt"},
            "ko": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/korean_PP-OCRv4_rec_infer.tar",
                       "dict_path": "forge/test/models/paddlepaddle/multimodal/paddleocr/cached_models/korean_dict.txt"},
            "ja": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/japan_PP-OCRv4_rec_infer.tar",
                   "dict_path": "forge/test/models/paddlepaddle/multimodal/paddleocr/cached_models/japan_dict.txt"},
        },
    },
    "v0": {
        "det": {
            "ch": {"url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar"},
            "en": {
                "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_ppocr_mobile_v2.0_det_infer.tar"
            },
        },
        "rec": {
            "ch": {"url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar",
                   "dict_path": "forge/test/models/paddlepaddle/multimodal/paddleocr/cached_models/ppocr_keys_v1.txt"},
            "en": {
                "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar",
                "dict_path": "forge/test/models/paddlepaddle/multimodal/paddleocr/cached_models/en_dict.txt"
            },
        },
    },
}

cache_dir = os.path.join("forge/test/models/paddlepaddle/multimodal/paddleocr", "cached_models")
os.makedirs(cache_dir, exist_ok=True)

class PaddleOCRVariant(Enum):
    PP_OCR = "v0"
    PP_OCRv4 = "v4"

class Language(Enum):
    CHINESE = "ch"
    ENGLISH = "en"
    KOREAN = "ko"
    JAPANESE = "ja"

def paddleocr(variant, language, image_path, results_path, cache_dir):
    # Fetch model
    det_url = model_urls[variant]["det"][language]["url"]
    rec_url = model_urls[variant]["rec"][language]["url"]
    dict_path = model_urls[variant]["rec"][language]["dict_path"]

    detection_model = fetch_paddle_model(det_url, cache_dir)
    recognition_model = fetch_paddle_model(rec_url, cache_dir)

    # Load sample
    image = cv2.imread(image_path)
    new_shapes = ((image.shape[1] // 32) * 32, (image.shape[0] // 32) * 32)
    resized_image = cv2.resize(image, (new_shapes))
    image = resized_image.transpose(2, 0, 1).astype("float32")
    inputs = [paddle.to_tensor([image])]

    # Detection - find boxes containing text
    pred = detection_model(*inputs)

    # Visualize prediction as heatmap
    heatmap = (pred[0, 0].numpy() * 255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(f"{results_path}/pred_heatmap.jpg", heatmap)

    bitmap = bitmap_from_probmap(pred)
    dest_height, dest_width = image.shape[1:]
    boxes, _ = polygons_from_bitmap(bitmap, dest_height=dest_height, dest_width=dest_width)

    # Visualize bitmap
    bitmap_visual = (bitmap * 255).astype("uint8")
    cv2.imwrite(f"{results_path}/bitmap.jpg", bitmap_visual)

    # Visualize polygons over image
    img = draw_boxes(resized_image, boxes)
    cv2.imwrite(f"{results_path}/det_clouds.jpg", img)

    # Visualize boxes over image
    box_cuts, img = cut_boxes(resized_image, boxes)
    cv2.imwrite(f"{results_path}/det_boxes.jpg", img)

    # Unify image sizes for recognition with compiled model
    padded_box_cuts = process_and_pad_images(box_cuts, img_size = (dest_height, dest_width))

    image_0 = padded_box_cuts[0]
    image_0 = image_0.transpose(2, 0, 1).astype("float32")/255.0
    image_0 = paddle.to_tensor([image_0])
    inputs = [image_0]

    # Compile model
    compiled_model = forge.compile(recognition_model, inputs)

    # Verify data on sample input
    verify(
        inputs,
        recognition_model,
        compiled_model
    )

    # Load dictionary
    with open(dict_path, "r", encoding="utf-8") as f:
        charset = [line.strip() for line in f.readlines()]
        charset.insert(0, '')
        charset.append('')

    # Recognition - recognize text in each box
    for i, box_image in enumerate(padded_box_cuts):
        # Save image of each box
        cv2.imwrite(f"{results_path}/box_{i}.jpg", box_image)

        # Preprocess box image
        box_image = box_image.transpose(2, 0, 1).astype("float32")/255.0
        box_image = paddle.to_tensor([box_image])

        # Run compiled recognition model
        output = compiled_model(box_image)[0].numpy()

        # Decode output
        pred = output.argmax(axis=2)[0]
        pred_str = "".join([charset[i] for i in pred])
        print(f"Predicted text for box {i}: {pred_str}")

if __name__ == "__main__":
    # Example usage:
    # python test_paddleocr.py --variant PP-OCRv4 --language en --image_path /path/to/image.jpg --results_path /path/to/results

    parser = argparse.ArgumentParser(description="Run PaddleOCR on an image.")
    parser.add_argument("--variant", type=str, required=True, choices=[v.value for v in PaddleOCRVariant],
                        help="PaddleOCR variant to use (e.g., PP-OCR, PP-OCRv4).")
    parser.add_argument("--language", type=str, required=True, choices=[l.value for l in Language],
                        help="Language to use for OCR (e.g., ch, en, ko, ja).")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--results_path", type=str, required=True, help="Path to save the results.")
    args = parser.parse_args()

    paddleocr(
        variant=args.variant,
        language=args.language,
        image_path=args.image_path,
        results_path=args.results_path,
        cache_dir=cache_dir
    )

 