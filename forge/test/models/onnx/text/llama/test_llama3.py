# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.models.llama.modeling_llama import (
    AttentionMaskConverter,
    Cache,
    LlamaModel,
    StaticCache,
)

import forge
from forge.verify.verify import verify

from forge.forge_property_utils import Framework, Source, Task
from test.models.models_utils import build_optimum_cli_command
from test.utils import download_model
import subprocess
import onnx

# Monkey Patching Casual Mask Update
def _update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool,
):
    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None

    # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    # to infer the attention mask.
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    using_static_cache = isinstance(past_key_values, StaticCache)

    # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
    if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask,
            inputs_embeds=input_tensor,
            past_key_values_length=past_seen_tokens,
            is_training=self.training,
        ):
            return None

    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    if using_static_cache:
        target_length = past_key_values.get_max_length()
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    if attention_mask is not None and attention_mask.dim() == 4:
        # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
        if attention_mask.max() != 0:
            raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        # return causal_mask
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0

            # Replace Implace Slice Update
            # causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            #     padding_mask, min_dtype
            # )

            if causal_mask.shape[-1] < mask_length:
                part_1 = causal_mask[:, :, :, :mask_length]
                part_2 = causal_mask[:, :, :, mask_length:]
                part_1 = part_1.masked_fill(padding_mask, min_dtype)
                causal_mask = torch.cat([part_1, part_2], dim=-1)
            else:
                causal_mask = causal_mask.masked_fill(padding_mask, min_dtype)

    if (
        self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type == "cuda"
        and not output_attentions
    ):
        # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask


LlamaModel._update_causal_mask = _update_causal_mask


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "meta-llama/Llama-3.1-8B",
            marks=pytest.mark.skip(reason="Segmentation fault"),
        ),
        pytest.param(
            "meta-llama/Llama-3.2-1B",
            marks=pytest.mark.skip(reason="Insufficient host DRAM to run this model"),
        ),
        pytest.param(
            "meta-llama/Llama-3.2-3B",
            marks=pytest.mark.skip(reason="Segmentation fault"),
        ),
        pytest.param(
            "meta-llama/Llama-3.1-8B-Instruct",
            marks=pytest.mark.skip(reason="Segmentation fault"),
        ),
        pytest.param(
            "meta-llama/Llama-3.2-1B-Instruct",
            marks=pytest.mark.skip(reason="Insufficient host DRAM to run this model"),
        ),
        pytest.param(
            "meta-llama/Llama-3.2-3B-Instruct",
            marks=pytest.mark.skip(reason="Segmentation fault"),
        ),
    ],
)
def test_llama3_causal_lm_onnx(forge_property_recorder, variant, tmp_path):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX, model="llama3", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    if variant in [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]:
        forge_property_recorder.record_group("red")
        forge_property_recorder.record_priority("P2")
    else:
        forge_property_recorder.record_group("generality")

    # Load model and tokenizer
    framework_model = download_model(AutoModelForCausalLM.from_pretrained, variant, use_cache=False, return_dict=False)
    framework_model.eval()
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare input
    input_prompt = "Hey how are you doing today?"
    if variant == "meta-llama/Llama-3.2-3B":
        inputs = tokenizer.encode_plus(
            input_prompt,
            return_tensors="pt",
            max_length=32,
            padding="max_length",
            truncation=True,
        )
    else:
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

    # Export model to ONNX
    onnx_path = f"{tmp_path}/model.onnx"
    command = build_optimum_cli_command(variant, tmp_path)
    subprocess.run(command, check=True)

    # Load framework model
    onnx_model = onnx.load(onnx_path)

    # passing model file instead of model proto due to size of the model(>2GB) - #https://github.com/onnx/onnx/issues/3775#issuecomment-943416925
    onnx.checker.check_model(onnx_path)
    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Compile model
    compiled_model = forge.compile(
        framework_model, inputs, forge_property_handler=forge_property_recorder, module_name=module_name
    )

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
        forge_property_handler=forge_property_recorder,
    )
