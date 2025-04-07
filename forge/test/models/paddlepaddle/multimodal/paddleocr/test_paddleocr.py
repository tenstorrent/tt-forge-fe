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

model_urls = {
    "PP-OCRv4": {
        "det": {
            "ch": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar"},
            "en": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_det_infer.tar"},
            "ml": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar"},
        },
        "rec": {
            "ch": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar"},
            "en": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar"},
            "korean": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/korean_PP-OCRv4_rec_infer.tar"},
            "japanese": {"url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/japan_PP-OCRv4_rec_infer.tar"},
        },
    },
    "PP-OCR": {
        "det": {
            "ch": {"url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar"},
            "en": {
                "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_ppocr_mobile_v2.0_det_infer.tar"
            },
        },
        "rec": {
            "ch": {"url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar"},
            "en": {
                "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar"
            },
        },
    },
}

cache_dir = os.path.join("forge/test/models/paddlepaddle/multimodal/paddleocr", "cached_models")
os.makedirs(cache_dir, exist_ok=True)


@pytest.mark.nightly
@pytest.mark.xfail()
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
    image = cv2.resize(image, (224, 224)).transpose(2, 0, 1).astype("float32")
    inputs = [paddle.to_tensor([image])]

    # Test framework model
    print(framework_model(inputs[0]))

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
