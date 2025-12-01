# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

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
from third_party.tt_forge_models.vilt.masked_lm.pytorch import (
    ModelLoader as MaskedLMLoader,
)
from third_party.tt_forge_models.vilt.masked_lm.pytorch import (
    ModelVariant as MaskedLMVariant,
)
from third_party.tt_forge_models.vilt.question_answering.pytorch import (
    ModelLoader as QuestionAnsweringLoader,
)
from third_party.tt_forge_models.vilt.question_answering.pytorch import (
    ModelVariant as QuestionAnsweringVariant,
)

from test.models.pytorch.multimodal.vilt.model_utils.model import (
    ViLtEmbeddingWrapper,
    ViltModelWrapper,
)


def generate_model_vilt_question_answering_hf_pytorch(variant):

    # Load model and inputs
    loader = QuestionAnsweringLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    encoding = loader.load_inputs(dtype_override=torch.bfloat16)
    model.config.return_dict = False

    # Wrapper
    text_vision_embedding_model = ViLtEmbeddingWrapper(model)
    vilt_model = ViltModelWrapper(model, task=Task.QA.short)

    embedding_output, attention_mask = text_vision_embedding_model(**encoding)

    return vilt_model, [embedding_output.detach().cpu(), attention_mask.detach().cpu().to(torch.float32)], loader


qa_variants = [QuestionAnsweringVariant.VQA]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", qa_variants)
def test_vilt_question_answering_hf_pytorch(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH, model=ModelArch.VILT, variant=variant, task=Task.QA, source=Source.HUGGINGFACE
    )

    framework_model, inputs, loader = generate_model_vilt_question_answering_hf_pytorch(variant)

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
    print("Predicted answer:", loader.decode_output(co_out))


def generate_model_vilt_maskedlm_hf_pytorch(variant):

    # Load model and inputs
    loader = MaskedLMLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    encoding = loader.load_inputs(dtype_override=torch.bfloat16)

    # prepare model and input
    model.config.return_dict = False
    text_vision_embedding_model = ViLtEmbeddingWrapper(model)
    vilt_model = ViltModelWrapper(model=model, task=Task.MASKED_LM.short, text_seq_len=encoding["input_ids"].shape[1])

    embedding_output, attention_mask = text_vision_embedding_model(**encoding)

    return vilt_model, [embedding_output.detach().cpu(), attention_mask.detach().cpu().to(torch.float32)], {}


mlm_variants = [MaskedLMVariant.MLM]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", mlm_variants)
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
