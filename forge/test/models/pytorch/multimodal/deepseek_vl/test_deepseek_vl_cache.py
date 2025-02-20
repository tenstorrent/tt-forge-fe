# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from loguru import logger
from transformers import AutoModelForCausalLM

import forge

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

        def forward(self, inputs_embeds, attention_mask=None, position_ids=None, *past_key_values):
            if past_key_values is not None:
                past_key_values = [past_key_values[i : i + 2] for i in range(0, len(past_key_values), 2)]
            else:
                past_key_values = None
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


def calculate_attention_mask_and_postion_ids(
    padded_past_key_values_seq_length, non_padding_past_key_values_seq_length, input_seq_length
):

    # Calculate attention mask
    attention_mask = torch.zeros(padded_past_key_values_seq_length + input_seq_length)
    attention_mask[:non_padding_past_key_values_seq_length] = 1
    attention_mask[-1] = 1
    attention_mask = attention_mask.unsqueeze(0)

    # Calculate position ids
    position_ids = torch.arange(
        non_padding_past_key_values_seq_length,
        input_seq_length + non_padding_past_key_values_seq_length,
        dtype=torch.long,
    )
    position_ids = position_ids.unsqueeze(0)

    return attention_mask, position_ids


@pytest.mark.parametrize("run_on_tt_device", [True])
def test_deepseek_vl_pytorch_cache(run_on_tt_device):

    framework_model, vl_gpt, tokenizer, inputs_embeds = generate_model_deepseek_vl_pytorch(
        "deepseek-ai/deepseek-vl-1.3b-base"
    )
    max_new_tokens = 512
    prefill_output = framework_model(inputs_embeds=inputs_embeds)
    next_token_logits = prefill_output[0][:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)

    generated_token_ids = []
    generated_token_ids.append(next_token.item())
    past_key_values_list = [[k, v] for k, v in prefill_output[1]]
    new_embedding = vl_gpt.language_model.get_input_embeddings()(next_token.unsqueeze(0))
    model_inputs = [new_embedding, past_key_values_list]
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
    model_inputs[1] = [tensor for sublist in model_inputs[1] for tensor in sublist]

    # key_tensor = [kv[0] for kv in model_inputs[1]]
    # value_tensor = [kv[1] for kv in model_inputs[1]]
    # key_tensor = torch.stack(key_tensor, dim=0)
    # value_tensor = torch.stack(value_tensor, dim=0)
    if run_on_tt_device:
        # Calculate attention mask and postion_ids
        # padded_past_key_values_seq_length = key_tensor[0].shape[-2]
        padded_past_key_values_seq_length = model_inputs[1][0].shape[-2]
        input_seq_length = model_inputs[0].shape[-2]
        attention_mask, position_ids = calculate_attention_mask_and_postion_ids(
            padded_past_key_values_seq_length, non_padding_seq_length, input_seq_length
        )

        # Compile the model
        attention_mask_rand = torch.randn(1, 1141)
        model_inputs_rand = [torch.randn(model_inputs[1][0].shape) for _ in range(len(model_inputs[1]))]
        # breakpoint()
        logger.info("Before compilation")
        logger.info(f"model_inputs[0] = {model_inputs[0].shape}")
        logger.info(f"attention_mask_rand = {attention_mask_rand.shape}")
        logger.info(f"position_ids = {position_ids.shape}")
        logger.info(f"model_inputs_rand[0] = {model_inputs_rand[0].shape}")
        compiled_model = forge.compile(
            framework_model,
            sample_inputs=[model_inputs[0], attention_mask_rand, position_ids, *model_inputs_rand],
            module_name="deepseek_vl",
        )

    for step in range(max_new_tokens):
        logger.info(f"step = {step}")
        non_padding_past_key_values_seq_length = non_padding_seq_length + step
        logger.info(f"non_padding_past_key_values_seq_length = {non_padding_past_key_values_seq_length}")
        padded_past_key_values_seq_length = model_inputs[1][0].shape[-2]
        logger.info(f"padded_past_key_values_seq_length = {padded_past_key_values_seq_length}")
        input_seq_length = model_inputs[0].shape[-2]

        # Run model inference
        attention_mask, position_ids = calculate_attention_mask_and_postion_ids(
            padded_past_key_values_seq_length, non_padding_past_key_values_seq_length, input_seq_length
        )
        model_outputs = framework_model(model_inputs[0], attention_mask, position_ids, *model_inputs[1])
        framework_output = [model_outputs[0]]
        for k, v in model_outputs[1]:
            framework_output.append(k)
            framework_output.append(v)
        if run_on_tt_device:
            # Run on TT device
            # tt_inputs = [model_inputs[0], attention_mask, position_ids, key_tensor]
            logger.info(f"Runtime")
            logger.info(f"model_inputs[0] = {model_inputs[0].shape}")
            logger.info(f"attention_mask = {attention_mask.shape}")
            logger.info(f"position_ids = {position_ids.shape}")
            logger.info(f"model_inputs_rand[0] = {model_inputs_rand[0].shape}")
            tt_output = compiled_model(model_inputs[0], attention_mask, position_ids, *model_inputs[1])
            tt_output = [tt_out.to("cpu") for tt_out in tt_output]

            # Validate TT result with Framework
            assert all(
                [
                    compare_with_golden(golden=fw_out, calculated=tt_out)
                    for fw_out, tt_out in zip(framework_output, tt_output)
                ]
            )

            logits = tt_output[0]
            past_key_values = tt_output[1:]

        else:
            logits = framework_output[0]
            past_key_values = framework_output[1:]
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        generated_token_ids.append(next_token_id.item())
        new_embedding = vl_gpt.language_model.get_input_embeddings()(next_token_id.unsqueeze(0))
        model_inputs = [new_embedding]
        model_inputs.append([past_key_values[idx : idx + 2] for idx in range(0, len(past_key_values), 2)])
        for idx in range(len(model_inputs[1])):
            model_inputs[1][idx][0][:, :, non_padding_past_key_values_seq_length, :] = model_inputs[1][idx][0][
                :, :, -1, :
            ]
            logger.info(f"shape of key_tensor = {model_inputs[1][idx][0].shape}")
            model_inputs[1][idx][0] = model_inputs[1][idx][0][:, :, :-1, :]
            logger.info(f"shape of key_tensor = {model_inputs[1][idx][0].shape}")
            model_inputs[1][idx][1][:, :, non_padding_past_key_values_seq_length, :] = model_inputs[1][idx][1][
                :, :, -1, :
            ]
            model_inputs[1][idx][1] = model_inputs[1][idx][1][:, :, :-1, :]
        model_inputs[1] = [tensor for sublist in model_inputs[1] for tensor in sublist]
    answer = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    print(answer)
