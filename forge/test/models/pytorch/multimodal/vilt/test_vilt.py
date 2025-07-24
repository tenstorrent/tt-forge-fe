# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from transformers import (
    ViltConfig,
    ViltForMaskedLM,
    ViltForQuestionAnswering,
    ViltProcessor,
)

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import VerifyConfig, verify

from test.models.pytorch.multimodal.vilt.model_utils.model import (
    ViLtEmbeddingWrapper,
    ViltModelWrapper,
)
from test.utils import download_model

text1 = "How many cats are there?"
text2 = "a bunch of cats laying on a [MASK]."


def get_image():
    input_image = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
    return Image.open(str(input_image))


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

    encoding = processor(get_image(), text1, return_tensors="pt")

    # Wrapper
    text_vision_embedding_model = ViLtEmbeddingWrapper(model)
    vilt_model = ViltModelWrapper(model, task=Task.QA.short)

    embedding_output, attention_mask = text_vision_embedding_model(**encoding)

    return vilt_model, [embedding_output.detach().cpu(), attention_mask.detach().cpu().to(torch.float32)], model


variants = ["dandelin/vilt-b32-finetuned-vqa"]


@pytest.mark.nightly
@pytest.mark.push
@pytest.mark.requires_token
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vilt_question_answering_hf_pytorch(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH, model=ModelArch.VILT, variant=variant, task=Task.QA, source=Source.HUGGINGFACE
    )

    framework_model, inputs, model = generate_model_vilt_question_answering_hf_pytorch(variant)

    framework_model.to(torch.bfloat16)
    inputs = [inputs[0].to(torch.bfloat16), inputs[1].to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.98)),
    )

    # Post processing
    logits = co_out[0]
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
    encoding = processor(get_image(), text2, return_tensors="pt")

    # Wrapper
    text_vision_embedding_model = ViLtEmbeddingWrapper(model)
    vilt_model = ViltModelWrapper(model=model, task=Task.MASKED_LM, text_seq_len=encoding["input_ids"].shape[1])

    embedding_output, attention_mask = text_vision_embedding_model(**encoding)

    return vilt_model, [embedding_output.detach().cpu(), attention_mask.detach().cpu().to(torch.float32)], {}


variants = ["dandelin/vilt-b32-mlm"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vilt_maskedlm_hf_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VILT,
        variant=variant,
        task=Task.MASKED_LM,
        source=Source.HUGGINGFACE,
    )

    framework_model, inputs, _ = generate_model_vilt_maskedlm_hf_pytorch(variant)

    framework_model.to(torch.bfloat16)
    inputs = [inputs[0].to(torch.bfloat16), inputs[1].to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.97)),
    )
