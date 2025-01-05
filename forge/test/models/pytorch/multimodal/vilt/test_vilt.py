# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import os
import forge
import requests
import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltForMaskedLM, ViltConfig
from test.models.pytorch.multimodal.vilt.utils.model import ViLtEmbeddingWrapper, ViltModelWrapper
from forge.verify.compare import compare_with_golden
from test.models.utils import build_module_name


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
    vilt_model = ViltModelWrapper(model, task="qa")

    embedding_output, attention_mask = text_vision_embedding_model(**encoding)

    return vilt_model, [embedding_output.detach().cpu(), attention_mask.detach().cpu().to(torch.float32)], {}


variants = ["dandelin/vilt-b32-finetuned-vqa"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vilt_question_answering_hf_pytorch(record_forge_property, variant):
    module_name = build_module_name(framework="pt", model="vilt", variant=variant, task="qa", source="hf")

    record_forge_property("module_name", module_name)

    model, inputs, _ = generate_model_vilt_question_answering_hf_pytorch(variant)
    compiled_model = forge.compile(model, sample_inputs=[inputs[0], inputs[1]], module_name=module_name)


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
    vilt_model = ViltModelWrapper(model=model, task="maskedlm", text_seq_len=encoding["input_ids"].shape[1])

    embedding_output, attention_mask = text_vision_embedding_model(**encoding)

    return vilt_model, [embedding_output.detach().cpu(), attention_mask.detach().cpu().to(torch.float32)], {}


variants = ["dandelin/vilt-b32-mlm"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vilt_maskedlm_hf_pytorch(record_forge_property, variant):
    module_name = build_module_name(framework="pt", model="vilt", variant=variant, task="mlm", source="hf")

    record_forge_property("module_name", module_name)

    model, inputs, _ = generate_model_vilt_maskedlm_hf_pytorch(variant)
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
    co_out = compiled_model(*inputs)
    fw_out = model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])
