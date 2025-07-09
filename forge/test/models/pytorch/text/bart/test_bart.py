# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# BART Demo Script - SQuADv1.1 QA
import pytest
from transformers import BartForSequenceClassification, BartTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right

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

from test.utils import download_model


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "facebook/bart-large-mnli",
        ),
    ],
)
def test_pt_bart_classifier(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.BART,
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    model = download_model(
        BartForSequenceClassification.from_pretrained, variant, torchscript=True, use_cache=False, return_dict=False
    )
    tokenizer = download_model(BartTokenizer.from_pretrained, variant, pad_to_max_length=True)
    hypothesis = "Most of Mrinal Sen's work can be found in European collections."
    premise = "Calcutta seems to be the only other production center having any pretensions to artistic creativity at all, but ironically you're actually more likely to see the works of Satyajit Ray or Mrinal Sen shown in Europe or North America than in India itself."

    # generate inputs
    inputs_dict = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        padding="max_length",
        max_length=256,
        truncation_strategy="only_first",
        return_tensors="pt",
    )
    decoder_input_ids = shift_tokens_right(
        inputs_dict["input_ids"], model.config.pad_token_id, model.config.decoder_start_token_id
    )
    inputs = [inputs_dict["input_ids"], inputs_dict["attention_mask"], decoder_input_ids]

    # Forge compile framework model
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    _, co_out = verify(
        inputs, model, compiled_model, verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95))
    )

    # co_out = model(*inputs)

    # logits = co_out[0]

    # # Select entailment and contradiction logits
    # entail_contradiction_logits = logits[:, [0, 2]]  # [contradiction, entailment]

    # # Convert to probabilities
    # probs = entail_contradiction_logits.softmax(dim=1)

    # # Probability that the label (hypothesis) is entailed by the premise
    # prob_label_is_true = probs[:, 1].item()

    # print(f"\n Entailment probability for hypothesis: {prob_label_is_true:.4f}\n")
