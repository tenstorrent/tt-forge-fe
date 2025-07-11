# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from third_party.tt_forge_models.tools.utils import get_file

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

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
@pytest.mark.xfail
def test_surya_ocr():

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.SURYAOCR,
        variant="default",
        task=Task.CV_OBJECT_DET,
        source=Source.GITHUB,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    # Load model
    framework_model = SuryaOCRWrapper()
    framework_model.eval()

    # Even after setting eval, some model parameters still had requires_grad=True, which led to a "RuntimeError: Cannot insert a Tensor that requires grad as a constant. Consider making it a parameter or input, or detaching the gradient"
    # The fix was to explicitly disable gradient tracking for all parameters in both the recognition and detection models.
    for name, param in framework_model.recognition_predictor.model.named_parameters():
        param.requires_grad = False

    for name, param in framework_model.detection_predictor.model.named_parameters():
        param.requires_grad = False

    # Prepare input
    image_file = get_file(IMAGE_URL)
    image = Image.open(str(image_file))

    image_tensor = preprocess_image(image)
    langs = [["en"]]
    language_tensor = torch.tensor([[LANGUAGE_MAP[lang] for lang in batch] for batch in langs], dtype=torch.int64)
    images = torch.stack([image_tensor])
    inputs = [images, language_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
