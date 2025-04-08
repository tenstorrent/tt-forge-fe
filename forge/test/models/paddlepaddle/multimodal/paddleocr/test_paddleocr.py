# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import paddle
import pytest
import cv2

import forge
from forge.config import CompilerConfig
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import fetch_paddle_model
from test.models.paddlepaddle.multimodal.paddleocr.det_post_processing import bitmap_from_probmap, boxes_from_bitmap, draw_boxes, cut_boxes, polygons_from_bitmap

model_urls = {
    "PP-OCRv4": {
        "det": {
            "ch": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar"},
            # "en": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_det_infer.tar"},
            # "ml": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar"},
        },
        "rec": {
            "ch": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar"},
            # "en": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar"},
            # "korean": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/korean_PP-OCRv4_rec_infer.tar"},
            # "japanese": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/japan_PP-OCRv4_rec_infer.tar"},
        },
    },
    # "PP-OCR": {
    #     "det": {
    #         "ch": {"url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar"},
    #         "en": {
    #             "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_ppocr_mobile_v2.0_det_infer.tar"
    #         },
    #     },
    #     "rec": {
    #         "ch": {"url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar"},
    #         "en": {
    #             "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar"
    #         },
    #     },
    # },
}

cache_dir = os.path.join("forge/test/models/paddlepaddle/multimodal/paddleocr", "cached_models")
os.makedirs(cache_dir, exist_ok=True)


@pytest.mark.nightly
# @pytest.mark.xfail()
@pytest.mark.parametrize(
    "variant,url",
    [
        (f"{var}_{task}_{lang}", details["url"])
        for var, tasks in model_urls.items()
        for task, langs in tasks.items()
        if task == "det"
        for lang, details in langs.items()
    ],
)
def test_paddleocr_det(forge_property_recorder, variant, url):
    # Record model details
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="paddleocr",
        variant=variant,
        source=Source.PADDLE,
        task=Task.SCENE_TEXT_RECOGNITION,
    )
    forge_property_recorder.record_model_name(module_name)
    forge_property_recorder.record_group("generality")

    # Fetch model
    framework_model = fetch_paddle_model(url, cache_dir)

    # Load sample
    image_path = "forge/test/models/paddlepaddle/multimodal/paddleocr/images/ch_text.jpg"
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (448, 448))
    image = resized_image.transpose(2, 0, 1).astype("float32")
    inputs = [paddle.to_tensor([image])]

    # Test framework model
    pred = framework_model(*inputs)
    bitmap = bitmap_from_probmap(pred)
    dest_height, dest_width = image.shape[1:]
    boxes, _ = boxes_from_bitmap(pred, bitmap, dest_height=dest_height, dest_width=dest_width)

    img = draw_boxes(resized_image, boxes)
    cv2.imwrite("forge/test/models/paddlepaddle/multimodal/paddleocr/images/det_result.jpg", img)

    # Compile model
    compiled_model = forge.compile(
        framework_model,
        inputs,
        forge_property_handler=forge_property_recorder,
        module_name=module_name,
    )

    # Verify data on sample input
    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
        forge_property_handler=forge_property_recorder,
    )


# @pytest.mark.xfail()
@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant,url",
    [
        (f"{var}_{task}_{lang}", details["url"])
        for var, tasks in model_urls.items()
        for task, langs in tasks.items()
        if task == "rec"
        for lang, details in langs.items()
    ],
)
def test_paddleocr_rec(forge_property_recorder, variant, url):
    # Record model details
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="paddleocr",
        variant=variant,
        source=Source.PADDLE,
        task=Task.SCENE_TEXT_RECOGNITION,
    )
    forge_property_recorder.record_model_name(module_name)
    forge_property_recorder.record_group("generality")

    # Fetch model
    framework_model = fetch_paddle_model(url, cache_dir)

    # Load sample
    image_path = "forge/test/models/paddlepaddle/multimodal/paddleocr/images/ch_text.jpg"
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 32)).transpose(2, 0, 1).astype("float32")/255.0
    inputs = [paddle.to_tensor([image])]

    # Test framework model
    output = framework_model(*inputs)
    
    # Compile model
    compiled_model = forge.compile(
        framework_model,
        inputs,
        forge_property_handler=forge_property_recorder,
        module_name=module_name,
    )

    # Verify data on sample input
    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
        forge_property_handler=forge_property_recorder,
    )

det_url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_ppocr_mobile_v2.0_det_infer.tar"
rec_url = "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar"
dict_path = "forge/test/models/paddlepaddle/multimodal/paddleocr/cached_models/en_dict.txt"
image_path = "forge/test/models/paddlepaddle/multimodal/paddleocr/images/testocr.png"

def test_full_ocr():
    # Fetch model
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
    cv2.imwrite("forge/test/models/paddlepaddle/multimodal/paddleocr/images/pred_heatmap.jpg", heatmap)

    bitmap = bitmap_from_probmap(pred)
    dest_height, dest_width = image.shape[1:]
    boxes, _ = polygons_from_bitmap(pred, bitmap, dest_height=dest_height, dest_width=dest_width)
    # Visualize bitmap
    bitmap_visual = (bitmap * 255).astype("uint8")
    cv2.imwrite("forge/test/models/paddlepaddle/multimodal/paddleocr/images/bitmap_visual.jpg", bitmap_visual)

    img = draw_boxes(resized_image, boxes)
    cv2.imwrite("forge/test/models/paddlepaddle/multimodal/paddleocr/images/det_result.jpg", img)

    box_cuts, img = cut_boxes(resized_image, boxes)
    # Visualize boxes
    cv2.imwrite("forge/test/models/paddlepaddle/multimodal/paddleocr/images/box_result.jpg", img)

    with open(dict_path, "r", encoding="utf-8") as f:
        charset = [line.strip() for line in f.readlines()]
        charset.insert(0, '')

    # Recognition - recognize text in each box
    for i, box_image in enumerate(box_cuts):
        cv2.imwrite(f"forge/test/models/paddlepaddle/multimodal/paddleocr/images/cut_result_{i}.jpg", box_image)
        box_image = box_image.transpose(2, 0, 1).astype("float32")/255.0
        box_image = paddle.to_tensor([box_image])
        output = recognition_model(box_image)
        pred = paddle.argmax(output, axis=2).numpy()[0]
        pred_str = "".join([charset[i] for i in pred])
        print(f"Predicted text for box {i}: {pred_str}")

