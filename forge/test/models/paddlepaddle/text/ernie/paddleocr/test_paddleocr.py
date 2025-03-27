import os
import paddle
import pytest
import cv2

import forge
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name

def test_paddleocr():
    dir_path = "forge/test/models/paddlepaddle/text/ernie/paddleocr"
    variant = "en_PP-OCRv3_rec"
    model_path = os.path.join(dir_path, variant, "inference")
    model = paddle.jit.load(model_path)
    
    image_path = 'forge/test/models/paddlepaddle/text/ernie/paddleocr/error.png'
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224)).transpose(2,0,1).astype('float32')
    inputs = [paddle.to_tensor([image])]
    
    compiled_model = forge.compile(model, inputs)
    verify(inputs, model, compiled_model)

    # compiled_model = 