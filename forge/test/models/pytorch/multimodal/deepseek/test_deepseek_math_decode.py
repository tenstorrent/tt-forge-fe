# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import forge
from forge.verify.compare import compare_with_golden


class DeepSeekWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None):
        # If past_key_values is provided, it means we are in the decoding phase
        if past_key_values is not None:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return outputs.logits, outputs.past_key_values


@pytest.mark.parametrize("run_on_tt_device", [False, True])
def test_deepseek_prefill_on_cpu_decode_on_tt_no_cache(run_on_tt_device):
    # Load DeepSeek model and tokenizer
    model_name = "deepseek-ai/deepseek-math-7b-instruct"
    model, tokenizer, input_ids = download_model_and_tokenizer(model_name)
    framework_model = DeepSeekWrapper(model)
    framework_model.eval()

    # Prepare input sentence using chat template
    messages = [
        {
            "role": "user",
            "content": "what is the integral of x^2 from 0 to 2?\nPlease reason step by step, and put your final answer within \\boxed{}.",
        }
    ]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    non_padding_seq_len = (input_ids != tokenizer.pad_token_id).sum(dim=1).item()
    padding_seq_len = input_ids.shape[1] - non_padding_seq_len

    # Run Prefill on CPU
    logits = framework_model(input_ids=input_ids)

    # Take the last non-padding token index in logits for the next token
    logits = logits[0]
    next_token_logits = logits[:, non_padding_seq_len - 1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)

    # Update the input_ids with predicted token from prefill in the first padding token index
    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

    non_padding_seq_len = (input_ids != tokenizer.pad_token_id).sum(dim=1).item()
    padding_seq_len = input_ids.shape[1] - non_padding_seq_len

    if run_on_tt_device:
        compiler_cfg = forge.config.CompilerConfig()

        # Compile the model on TT
        compiled_model = forge.compile(framework_model, sample_inputs=[input_ids], compiler_cfg=compiler_cfg)

    # Run decode stage on TT device and generate tokens by appending predicted token into sequence of input tokens
    # until a specified maximum number of new tokens is reached or an end-of-sequence token is encountered.
    max_new_tokens = 200  # Set a reasonable limit for new tokens
    for idx in range(max_new_tokens):
        # CPU Inference
        logits, past_key_values = framework_model(input_ids=input_ids)

        if run_on_tt_device:
            model_inputs = [input_ids]
            # Run on TT device
            tt_output = compiled_model(*model_inputs)
            tt_output = [tt_out.to("cpu") for tt_out in tt_output]

            # Validate TT result with Framework
            assert all(
                [
                    compare_with_golden(golden=fw_out, calculated=tt_out)
                    for fw_out, tt_out in zip([logits, past_key_values], tt_output)
                ]
            )

            logits = tt_output[0]

        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        if next_token == tokenizer.eos_token_id:
            break

        # Append the new token to the input_ids
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

    # Generated text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print("generated_text=", generated_text)


def download_model_and_tokenizer(model_name, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.generation_config.use_cache = kwargs.get("use_cache", False)

    # Prepare input sentence
    messages = kwargs.get(
        "messages",
        [
            {
                "role": "user",
                "content": "what is the integral of x^2 from 0 to 2?\nPlease reason step by step, and put your final answer within \\boxed{}.",
            }
        ],
    )
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    return model, tokenizer, input_ids
