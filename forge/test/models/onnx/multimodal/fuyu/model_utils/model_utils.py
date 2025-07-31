# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


def generate_no_cache(max_new_tokens, model, inputs, seq_len, tokenizer):
    current_pos = seq_len

    for _ in range(max_new_tokens):
        logits = model(**inputs)

        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        next_token_logits = logits[:, current_pos - 1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        # Update input_ids and attention_mask
        inputs["input_ids"][:, current_pos] = next_token_id
        inputs["attention_mask"][:, current_pos] = 1

        current_pos += 1

    valid_tokens = inputs["input_ids"][:, seq_len:current_pos].view(-1).tolist()
    answer = tokenizer.decode(valid_tokens, skip_special_tokens=True)
    return answer
