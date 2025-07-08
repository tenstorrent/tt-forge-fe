# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def load_model(model_path="openlm-research/open_llama_3b", **kwargs):
    # Default config values
    config = LlamaConfig.from_pretrained(model_path)

    # Use defaults or values from kwargs
    config.return_dict = kwargs.get("return_dict", False)
    config.use_cache = kwargs.get("use_cache", False)
    config.output_attentions = kwargs.get("output_attentions", False)
    config.output_hidden_states = kwargs.get("output_hidden_states", False)
    if "num_hidden_layers" in kwargs and kwargs["num_hidden_layers"] is not None:
        config.num_hidden_layers = kwargs["num_hidden_layers"]

    # Load the model
    framework_model = LlamaForCausalLM.from_pretrained(model_path, config=config)
    framework_model.eval()

    use_lora = kwargs.get("use_lora", False)
    if use_lora:
        lora_r = kwargs.get("lora_r", 4)
        lora_alpha = kwargs.get("lora_alpha", 8)
        # Applying LoRA to the last half of all layers due to memory constraints, this should be configurable
        ltt = range(framework_model.config.num_hidden_layers // 2, framework_model.config.num_hidden_layers)
        lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, task_type="CAUSAL_LM", layers_to_transform=ltt)
        framework_model = get_peft_model(framework_model, lora_config)

    # Using AutoTokenizer for default tokenizers for both openllama and llama 3.2
    use_fast = kwargs.get("use_fast", True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)

    return framework_model, tokenizer


from datasets import load_dataset
from string import Template


# TRAIN_PROMPT_TEMPLATE = Template(
#     """
# Your task is to perform binary sentiment analysis and determine whether the sentiment of the review is negative or positive.
# Output should be in the valid json format: {'label': sentiment_value}.
# Review: $input
# Output:
# """
# )

TRAIN_PROMPT_TEMPLATE = Template("""Review: $input\nOutput:""")

LBL2VALUE = {0: "negative", 1: "positive"}


def load_tokenized_data(dataset_id, tokenizer, **kwargs):
    dataset = load_dataset(dataset_id)
    max_length = kwargs.get("max_length", 128)

    def _apply_template(examples):
        examples["text"] = [TRAIN_PROMPT_TEMPLATE.substitute(input=sentence) for sentence in examples["sentence"]]
        return examples

    def _tokenize_function(examples):
        tokenized_batch = tokenizer(examples["text"], padding="max_length", max_length=max_length, truncation=True)

        expected_output = [
            txt + f" {{'label': '{LBL2VALUE[lbl]}'}}" for txt, lbl in zip(examples["text"], examples["label"])
        ]
        tokenized_lbls = tokenizer(expected_output, padding="max_length", max_length=max_length, truncation=True)
        tokenized_batch["labels"] = tokenized_lbls["input_ids"]

        return tokenized_batch

    train_set = dataset["train"].map(_apply_template, batched=True)
    tokenized_train_set = train_set.map(_tokenize_function, batched=True)
    tokenized_train_set.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    sample_size = kwargs.get("sample_size", None)
    if sample_size:
        return tokenized_train_set.select(range(sample_size))
    return tokenized_train_set


def batchify(data, batch_size):
    batches = []
    total_size = len(data)
    num_full_batches = total_size // batch_size

    for i in range(num_full_batches):
        batch = data.select(range(i * batch_size, (i + 1) * batch_size))
        batches.append({
            "input_ids": [k["input_ids"] for k in batch],
            "attention_mask": [k["attention_mask"] for k in batch],
            "labels": [k["labels"] for k in batch],
        })


    return batches