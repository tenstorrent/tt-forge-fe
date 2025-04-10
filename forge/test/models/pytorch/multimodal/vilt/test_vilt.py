# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import requests
import torch
from PIL import Image
from transformers import (
    ViltConfig,
    ViltForMaskedLM,
    ViltForQuestionAnswering,
    ViltProcessor,
)

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.multimodal.vilt.utils.model import (
    ViLtEmbeddingWrapper,
    ViltModelWrapper,
)
from test.utils import download_model

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

text1 = "How many cats are there?"
text2 = "a bunch of cats laying on a [MASK]."


def generate_model_vilt_question_answering_hf_pytorch(variant):
    # Set model configurations
    config = ViltConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config = ViltConfig(**config_dict)

    # Load model and processor from HuggingFace
    processor = download_model(ViltProcessor.from_pretrained, variant)
    model = download_model(ViltForQuestionAnswering.from_pretrained, variant, config=config)
    model.eval()

    encoding = processor(image, text1, return_tensors="pt")

    # Wrapper
    text_vision_embedding_model = ViLtEmbeddingWrapper(model)
    vilt_model = ViltModelWrapper(model, task=Task.QA)

    embedding_output, attention_mask = text_vision_embedding_model(**encoding)

    return vilt_model, [embedding_output.detach().cpu(), attention_mask.detach().cpu().to(torch.float32)], model


variants = ["dandelin/vilt-b32-finetuned-vqa"]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vilt_question_answering_hf_pytorch(forge_property_recorder, variant):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="vilt", variant=variant, task=Task.QA, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    framework_model, inputs, model = generate_model_vilt_question_answering_hf_pytorch(variant)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Inference
    output = compiled_model(*inputs)

    # Post processing
    logits = output[0]
    idx = logits.argmax(-1).item()
    print("Predicted answer:", model.config.id2label[idx])


def generate_model_vilt_maskedlm_hf_pytorch(variant):
    # Set model configurations
    config = ViltConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config = ViltConfig(**config_dict)

    # Load model and processor from HuggingFace
    processor = download_model(ViltProcessor.from_pretrained, variant)
    model = download_model(ViltForMaskedLM.from_pretrained, variant, config=config)
    model.eval()

    # prepare inputs
    encoding = processor(image, text2, return_tensors="pt")

    # Wrapper
    text_vision_embedding_model = ViLtEmbeddingWrapper(model)
    vilt_model = ViltModelWrapper(model=model, task=Task.MASKED_LM, text_seq_len=encoding["input_ids"].shape[1])

    embedding_output, attention_mask = text_vision_embedding_model(**encoding)

    return vilt_model, [embedding_output.detach().cpu(), attention_mask.detach().cpu().to(torch.float32)], {}


variants = ["dandelin/vilt-b32-mlm"]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vilt_maskedlm_hf_pytorch(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="vilt", variant=variant, task=Task.MASKED_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    framework_model, inputs, _ = generate_model_vilt_maskedlm_hf_pytorch(variant)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
