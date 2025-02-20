# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import AutoModelForCausalLM

from test.models.pytorch.multimodal.deepseek_vl.utils.models.modeling_vlm import (
    MultiModalityCausalLM,
    VLChatProcessor,
    load_pil_images,
)


def generate_model_deepseek_vl_pytorch(variant):
    model_path = variant
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, model, max_new_tokens=512):
            super().__init__()
            self.model = model
            self.eos_token_id = tokenizer.eos_token_id
            self.bos_token_id = tokenizer.bos_token_id
            self.pad_token_id = tokenizer.pad_token_id
            self.max_new_tokens = max_new_tokens

        def forward(self, inputs_embeds, attention_mask=None, position_ids=None, past_key_values=None):
            output = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                pad_token_id=self.pad_token_id,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            return output.logits, output.past_key_values

    framework_model = Wrapper(vl_gpt)

    conversation = [
        {
            "role": "User",
            "content": "<image_placeholder>Describe each stage of this image.",
            "images": ["forge/test/models/pytorch/multimodal/deepseek_vl/image/training_pipelines.jpg"],
        },
        {"role": "Assistant", "content": ""},
    ]

    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True)
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    return framework_model, vl_gpt, tokenizer, inputs_embeds


def generate(model, tokenizer, vl_gpt, inputs_embeds, max_new_tokens=2048):

    prefill_output = model(inputs_embeds=inputs_embeds)
    next_token_logits = prefill_output[0][:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)

    # generated_tokens = inputs.input_ids
    # generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)

    past_key_values_list = [[k, v] for k, v in prefill_output[1]]

    model_inputs = [next_token.unsqueeze(0), past_key_values_list]
    non_padding_seq_length = past_key_values_list[0][0].shape[-2]
    for idx, (k, v) in enumerate(model_inputs[1]):
        model_inputs[1][idx][0] = torch.cat(
            [
                k,
                torch.zeros(k.shape[-4], k.shape[-3], max_new_tokens, k.shape[-1]).to(k.dtype),
            ],
            dim=-2,
        )
        model_inputs[1][idx][1] = torch.cat(
            [
                v,
                torch.zeros(v.shape[-4], v.shape[-3], max_new_tokens, v.shape[-1]).to(k.dtype),
            ],
            dim=-2,
        )
    generated_token_ids = []

    for step in range(max_new_tokens):
        non_padding_past_key_values_seq_length = non_padding_seq_length + step
        padded_past_key_values_seq_length = model_inputs[1][0][0].shape[-2]
        input_seq_length = model_inputs[0].shape[-1]

        # Run model inference
        logits, past_key_values = model(inputs_embeds, past_key_values=model_inputs[1])

        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        generated_token_ids.append(next_token_id.item())
        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(next_token_id.unsqueeze(0))
        model_inputs = [next_token_id.unsqueeze(0)]
        model_inputs.append([past_key_values[idx : idx + 2] for idx in range(0, len(past_key_values), 2)])
        for idx in range(len(model_inputs[1])):
            model_inputs[1][idx][0][:, :, non_padding_past_key_values_seq_length, :] = model_inputs[1][idx][0][
                :, :, -1, :
            ]
            model_inputs[1][idx][0] = model_inputs[1][idx][0][:, :, :-1, :]
            model_inputs[1][idx][1][:, :, non_padding_past_key_values_seq_length, :] = model_inputs[1][idx][1][
                :, :, -1, :
            ]
            model_inputs[1][idx][1] = model_inputs[1][idx][1][:, :, :-1, :]

    answer = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return answer


@pytest.mark.parametrize("variant", ["deepseek-ai/deepseek-vl-1.3b-base"])
def test_deepseek_vl_no_cache_cpu_pytorch(record_forge_property, variant):

    framework_model, vl_gpt, tokenizer, inputs_embeds = generate_model_deepseek_vl_pytorch(variant)
    answer = generate(framework_model, tokenizer, vl_gpt, inputs_embeds, max_new_tokens=2048)
