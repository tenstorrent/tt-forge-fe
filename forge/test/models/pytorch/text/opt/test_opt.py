# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from third_party.tt_forge_models.opt.causal_lm.pytorch.loader import (
    ModelLoader as CausalLMLoader,
)
from third_party.tt_forge_models.opt.causal_lm.pytorch.loader import (
    ModelVariant as CausalLMVariant,
)
from third_party.tt_forge_models.opt.qa.pytorch.loader import ModelLoader as QALoader
from third_party.tt_forge_models.opt.qa.pytorch.loader import ModelVariant as QAVariant
from third_party.tt_forge_models.opt.sequence_classification.pytorch.loader import (
    ModelLoader as SequenceClassificationLoader,
)
from third_party.tt_forge_models.opt.sequence_classification.pytorch.loader import (
    ModelVariant as SequenceClassificationVariant,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify


class OptModelWrapper(torch.nn.Module):
    def __init__(self, model, text_embedding):
        super().__init__()
        self.model = model
        self.text_embedding = text_embedding

    def forward(self, input_ids, attention_mask):
        inputs_embeds = self.text_embedding(input_ids)
        past_key_values_length = 0
        causal_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_ids.shape, inputs_embeds, past_key_values_length
        )
        position_ids = torch.cumsum(attention_mask, dim=1)
        position_ids = (position_ids * attention_mask - 1).long()
        position_ids = position_ids[:, past_key_values_length:]
        outputs = self.model(
            attention_mask=causal_attention_mask, inputs_embeds=inputs_embeds, position_ids=position_ids
        )
        if isinstance(outputs, (CausalLMOutputWithPast, SequenceClassifierOutputWithPast)):
            return outputs.logits
        elif isinstance(outputs, QuestionAnsweringModelOutput):
            return outputs.start_logits, outputs.end_logits
        else:
            return outputs


variants = [
    pytest.param(
        CausalLMVariant.OPT_125M,
    ),
    pytest.param(
        CausalLMVariant.OPT_350M,
    ),
    pytest.param(
        CausalLMVariant.OPT_1_3B,
        marks=[pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-mlir/issues/4174")],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_opt_causal_lm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.OPT,
        variant=variant.value,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Load model and inputs using model loader
    model_loader = CausalLMLoader(variant)
    framework_model = model_loader.load_model()
    framework_model = OptModelWrapper(framework_model, framework_model.model.decoder.embed_tokens)
    inputs = model_loader.load_inputs()

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        inputs,
        module_name,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants = [
    QAVariant.OPT_125M,
    QAVariant.OPT_350M,
    pytest.param(QAVariant.OPT_1_3B, marks=[pytest.mark.xfail]),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_opt_qa(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH, model=ModelArch.OPT, variant=variant.value, task=Task.QA, source=Source.HUGGINGFACE
    )

    # Load model and inputs using model loader
    # NOTE: These model variants are pre-trained only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.
    model_loader = QALoader(variant)
    framework_model = model_loader.load_model()
    framework_model = OptModelWrapper(framework_model, framework_model.model.decoder.embed_tokens)
    inputs = model_loader.load_inputs()

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        inputs,
        module_name,
    )

    pcc = 0.99
    if variant in [QAVariant.OPT_125M, QAVariant.OPT_1_3B]:
        pcc = 0.95

    # Model Verification
    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))


variants = [
    SequenceClassificationVariant.OPT_125M,
    SequenceClassificationVariant.OPT_350M,
    SequenceClassificationVariant.OPT_1_3B,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_opt_sequence_classification(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.OPT,
        variant=variant.value,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load model and inputs using model loader
    # NOTE: These model variants are pre-trained only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.
    model_loader = SequenceClassificationLoader(variant)
    framework_model = model_loader.load_model()
    framework_model = OptModelWrapper(framework_model, framework_model.model.decoder.embed_tokens)
    inputs = model_loader.load_inputs()

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        inputs,
        module_name,
    )

    # Model Verification and inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    model_loader.decode_output(co_out)
