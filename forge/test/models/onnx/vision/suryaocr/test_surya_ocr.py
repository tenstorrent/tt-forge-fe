# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from io import BytesIO

import pytest
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from surya.detection import DetectionPredictor

from surya.recognition import RecognitionPredictor
from test.models.models_utils import __call__without_post_processing, batch_recognition_without_post_processing

import forge
from forge.verify.verify import verify
import onnx

RecognitionPredictor.__call__ = __call__without_post_processing
RecognitionPredictor.batch_recognition = batch_recognition_without_post_processing


# from test.models.utils import Framework, Source, Task, build_module_name

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

        return ocr_results


@pytest.mark.nightly
def test_surya_ocr():

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

    # cpu inf
    with torch.no_grad():
        op = framework_model(images, language_tensor)

    print("op", op)
    # framework_model.eval()

    for name, param in framework_model.recognition_predictor.model.named_parameters():
        # print(name, param.requires_grad)
        param.requires_grad = False

    for name, param in framework_model.detection_predictor.model.named_parameters():
        # print(name, param.requires_grad)
        param.requires_grad = False

    print(" ============================ after req grad false ============================")

    for name, param in framework_model.recognition_predictor.model.named_parameters():
        print(name, param.requires_grad)

    for name, param in framework_model.detection_predictor.model.named_parameters():
        print(name, param.requires_grad)

    onnx_path = "forge/test/models/onnx/vision/suryaocr/surya_ocr.onnx"
    torch.onnx.export(framework_model, sample_inputs, onnx_path, opset_version=17)

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
