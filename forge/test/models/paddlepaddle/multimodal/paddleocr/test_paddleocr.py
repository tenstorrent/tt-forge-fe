import os
import paddle
import pytest
import cv2

import forge
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import fetch_paddle_model

model_urls = {
                #"en_PP-OCRv4_rec":"https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
              "en_PP-OCRv3_det":"https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar"
              }

cache_dir = os.path.join("forge/test/models/paddlepaddle/multimodal/paddleocr", "cached_models")
os.makedirs(cache_dir, exist_ok=True)

@pytest.mark.parametrize("variant,url", model_urls.items())
def test_paddleocr(forge_property_recorder,variant,url):
    # Record model details
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="paddleocr",
        variant=variant,
        source=Source.PADDLE,
        task=Task.SCENE_TEXT_RECOGNITION,
    )
    forge_property_recorder.record_model_name(module_name)

    framework_model = fetch_paddle_model(url, cache_dir)

    image_path = 'forge/test/models/paddlepaddle/multimodal/paddleocr/test_image.png'
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224)).transpose(2,0,1).astype('float32')
    inputs = [paddle.to_tensor([image])]

    print(framework_model(inputs[0]))

    # Compile model
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    # Verify data on sample input
    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
        forge_property_handler=forge_property_recorder,
   )