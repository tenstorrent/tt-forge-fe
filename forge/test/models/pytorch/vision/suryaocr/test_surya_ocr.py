# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import subprocess

# Install dependencies to run the surya_ocr model, the surya_ocr PYPI requires torch<3.0.0,>=2.5.1, but the env has torch 2.1.0+cpu.cxx11.abi which is incompatible. Hence we are installing it separately.
subprocess.run(["pip", "install", "surya-ocr", "--no-deps"])
subprocess.run(["pip", "install", "pydantic"])

from io import BytesIO

import pytest
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name

IMAGE_URL = "https://raw.githubusercontent.com/VikParuchuri/surya/master/static/images/excerpt_text.png"

LANGUAGE_MAP = {"en": 0, "fr": 1, "de": 2}


def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    return transform(image)


class SuryaOCRWrapper(nn.Module):
    def __init__(self):
        super(SuryaOCRWrapper, self).__init__()
        self.recognition_predictor = RecognitionPredictor()
        self.detection_predictor = DetectionPredictor()

    def forward(self, images_tensor, languages_tensor):
        batch_size = images_tensor.shape[0]
        images_list = [transforms.ToPILImage()(images_tensor[i]) for i in range(batch_size)]
        language_indices = languages_tensor.tolist()
        languages_list = [[list(LANGUAGE_MAP.keys())[i] for i in batch] for batch in language_indices]

        # Get OCR results
        ocr_results = self.recognition_predictor(images_list, languages_list, self.detection_predictor)

        text_confidence_list = []
        for batch in ocr_results:
            for result in batch.text_lines:
                text_confidence_list.append((result.text, result.confidence))

        return text_confidence_list


@pytest.mark.xfail(reason="Unknown output type: <class 'str'>")
@pytest.mark.nightly
def test_surya_ocr(record_forge_property):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="surya_ocr",
        variant="default",
        task=Task.OPTICAL_CHARACTER_RECOGNITION,
        source=Source.GITHUB,
    )

    # Record Forge Property
    record_forge_property("group", "priority_1")
    record_forge_property("tags.model_name", module_name)

    response = requests.get(IMAGE_URL)
    image = Image.open(BytesIO(response.content))
    image_tensor = preprocess_image(image)

    langs = [["en"]]
    language_tensor = torch.tensor([[LANGUAGE_MAP[lang] for lang in batch] for batch in langs], dtype=torch.int64)

    images = torch.stack([image_tensor])

    # Load model
    framework_model = SuryaOCRWrapper()

    # Load input
    sample_inputs = (images, language_tensor)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=sample_inputs, module_name="surya_ocr")

    # Model Verification
    verify(sample_inputs, framework_model, compiled_model)
