# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import (
    AutoTokenizer,
    Cache,
    Phi3Config,
    Phi3ForCausalLM,
    Phi3ForSequenceClassification,
    Phi3ForTokenClassification,
    Phi3Model,
    SlidingWindowCache,
)

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify


def _prepare_4d_causal_attention_mask_with_cache_position(
    self,
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_position: torch.Tensor,
    batch_size: int,
    config: Phi3Config,
    past_key_values: Cache,
):

    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        if config.sliding_window is not None:
            # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
            # the check is needed to verify is current checkpoint was trained with sliding window or not
            if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                sliding_attend_mask = torch.arange(target_length, device=device) <= (
                    cache_position.reshape(-1, 1) - config.sliding_window
                )
                diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
        causal_mask *= diagonal_attend_mask
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.shape[-1] > target_length:
                attention_mask = attention_mask[:, :target_length]
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0

            # Replace Implace Slice Update
            # causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            #     padding_mask, min_dtype
            # )
            if causal_mask.shape[-1] > mask_length:
                part_1 = causal_mask[:, :, :, :mask_length]
                part_2 = causal_mask[:, :, :, mask_length:]
                part_1 = part_1.masked_fill(padding_mask, min_dtype)
                causal_mask = torch.cat([part_1, part_2], dim=-1)
            else:
                causal_mask = causal_mask.masked_fill(padding_mask, min_dtype)
    return causal_mask


Phi3Model._prepare_4d_causal_attention_mask_with_cache_position = _prepare_4d_causal_attention_mask_with_cache_position


variants = ["microsoft/phi-3-mini-4k-instruct"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi3_causal_lm(forge_property_recorder, variant):
    pytest.skip("Insufficient host DRAM to run this model (requires a bit more than 38 GB)")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="phi3", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    if variant in ["microsoft/phi-3-mini-4k-instruct"]:
        forge_property_recorder.record_group("red")
    else:
        forge_property_recorder.record_group("generality")

    # Phi3Config from pretrained variant, disable return_dict and caching.
    config = Phi3Config.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = Phi3Config(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    framework_model = Phi3ForCausalLM.from_pretrained(variant, trust_remote_code=True, config=config)
    framework_model.eval()

    # input_prompt
    input_prompt = "Africa is an emerging economy because"

    # Tokenize input
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=256,
        pad_to_max_length=True,
        truncation=True,
    )

    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    inputs = [input_ids, attn_mask]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi3_token_classification(forge_property_recorder, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="phi3",
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Phi3Config from pretrained variant, disable return_dict and caching.
    config = Phi3Config.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = Phi3Config(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)

    framework_model = Phi3ForTokenClassification.from_pretrained(variant, trust_remote_code=True, config=config)
    framework_model.eval()

    # input_prompt
    input_prompt = "HuggingFace is a company based in Paris and New York"

    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt")

    inputs = [inputs["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name, forge_property_handler=forge_property_recorder)

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi3_sequence_classification(forge_property_recorder, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="phi3",
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Phi3Config from pretrained variant, disable return_dict and caching.
    config = Phi3Config.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config_dict["pad_token_id"] = None
    config = Phi3Config(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)
    framework_model = Phi3ForSequenceClassification.from_pretrained(variant, trust_remote_code=True, config=config)
    framework_model.eval()

    # input_prompt
    input_prompt = "the movie was great!"

    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt")
    inputs = [inputs["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name, forge_property_handler=forge_property_recorder)

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
