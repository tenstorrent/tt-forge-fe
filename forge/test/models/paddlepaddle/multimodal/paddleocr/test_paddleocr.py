# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import paddle
import pytest
import cv2

import forge
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task

from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker

from test.utils import fetch_paddle_model
from test.models.paddlepaddle.multimodal.paddleocr.utils import (
    get_boxes_from_pred,
    process_and_pad_images,
    fetch_img_and_charset,
)

model_urls = {
    "v4": {
        "det": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar",
        "rec": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar",
    }
}

image_url = "https://raw.githubusercontent.com/ycdhqzhiai/PaddleOCR-demo/main/imgs/11.jpg"
dict_url = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/ppocr_keys_v1.txt"

cache_dir = os.path.join("forge/test/models/paddlepaddle/multimodal/paddleocr", "cached_models")
os.makedirs(cache_dir, exist_ok=True)


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant,det_url,rec_url", [(variant, urls["det"], urls["rec"]) for variant, urls in model_urls.items()]
)
@pytest.mark.parametrize("image_url", [image_url])
@pytest.mark.parametrize("cache_dir", [cache_dir])
@pytest.mark.parametrize("dict_url", [dict_url])
# Uncomment if you want to save results
# @pytest.mark.parametrize("results_path", ["forge/test/models/paddlepaddle/multimodal/paddleocr/results"])
def test_paddleocr_det_on_cpu_rec_on_tt(
    forge_property_recorder, variant, det_url, rec_url, dict_url, image_url, cache_dir, results_path=None
):
    # Record model details
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PADDLE,
        model="paddleocr",
        variant=f"{variant}_det_on_cpu_rec_on_tt",
        source=Source.PADDLE,
        task=Task.SCENE_TEXT_RECOGNITION,
    )

    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Fetch model
    detection_model = fetch_paddle_model(det_url, cache_dir)
    recognition_model = fetch_paddle_model(rec_url, cache_dir)

    # Fetch image and dictionary
    image, charset = fetch_img_and_charset(image_url, dict_url)

    # Prepare inputs
    new_shapes = ((image.shape[1] // 32) * 32, (image.shape[0] // 32) * 32)
    resized_image = cv2.resize(image, (new_shapes))
    image = resized_image.transpose(2, 0, 1).astype("float32")
    inputs = [paddle.to_tensor([image])]

    # Compile model
    compiled_detection_model = forge.compile(
        detection_model, inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Detection - find boxes containing text
    pred = compiled_detection_model(*inputs)[0]

    box_cuts = get_boxes_from_pred(pred, image, resized_image, results_path=results_path)

    # Unify image sizes for recognition with compiled model
    padded_box_cuts = process_and_pad_images(box_cuts, img_size=image.shape[1:])

    image_0 = padded_box_cuts[0]
    image_0 = image_0.transpose(2, 0, 1).astype("float32") / 255.0
    image_0 = paddle.to_tensor([image_0])
    inputs = [image_0]

    # Compile model
    compiled_recognition_model = forge.compile(
        recognition_model, inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Recognition - recognize text in each box
    for i, box_image in enumerate(padded_box_cuts):
        if results_path:
            # Save image of each box
            cv2.imwrite(f"{results_path}/box_{i}.jpg", box_image)

        # Preprocess box image
        box_image = box_image.transpose(2, 0, 1).astype("float32") / 255.0
        box_image = paddle.to_tensor([box_image])

        # Run and verify compiled recognition model
        _, co_output = verify(
            [box_image],
            recognition_model,
            compiled_recognition_model,
            VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.8)),
            forge_property_handler=forge_property_recorder,
        )

        output = co_output[0].numpy()

        # Decode output
        pred = output.argmax(axis=2)[0]
        pred_str = "".join([charset[i] for i in pred])
        print(f"Predicted text for box {i}: {pred_str}")



