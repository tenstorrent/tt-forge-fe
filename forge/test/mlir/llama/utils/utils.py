# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from string import Template
import torch
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


def load_model(model_path="openlm-research/open_llama_3b", **kwargs):
    # Default config values
    config = LlamaConfig.from_pretrained(model_path)

    # Use defaults or values from kwargs
    config.return_dict = kwargs.get("return_dict", False)
    config.use_cache = kwargs.get("use_cache", False)
    config.output_attentions = kwargs.get("output_attentions", False)
    config.output_hidden_states = kwargs.get("output_hidden_states", False)
    if "num_hidden_layers" in kwargs:
        config.num_hidden_layers = kwargs["num_hidden_layers"]

    # Load the model
    framework_model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", config=config)
    framework_model = framework_model.to(dtype=torch.bfloat16)
    framework_model.eval()

    use_lora = kwargs.get("use_lora", False)
    if use_lora:
        lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
        framework_model = get_peft_model(framework_model, lora_config)
        framework_model.print_trainable_parameters()

    # Using AutoTokenizer for default tokenizers for both openllama and llama 3.2
    use_fast = kwargs.get("use_fast", True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)

    return framework_model, tokenizer


TRAIN_PROMPT_TEMPLATE = Template(
    """
Your task is to perform binary sentiment analysis and determine whether the sentiment of the review is negative or positive.
Output should be in the valid json format: {'label': sentiment_value}.

Review: $input

Output:
"""
)

LBL2VALUE = {0: "negative", 1: "positive"}


def load_tokenized_data(dataset_id, tokenizer, **kwargs):
    dataset = load_dataset(dataset_id)
    max_length = kwargs.get("max_length", 128)

    def _apply_template(example):
        example["text"] = TRAIN_PROMPT_TEMPLATE.substitute(input=example["sentence"])
        return example

    def _tokenize_function(example: dict):
        tokenized_batch = tokenizer(example["text"], padding="max_length", max_length=max_length, truncation=True)

        expected_output = example["text"] + f"Output: {{'label': '{LBL2VALUE[example['label']]}'}}"
        tokenized_lbls = tokenizer(expected_output, padding="max_length", max_length=max_length, truncation=True)
        tokenized_batch["labels"] = tokenized_lbls["input_ids"]

        return tokenized_batch

    train_set = dataset["train"].map(_apply_template, fn_kwargs={"mode": "train"})
    tokenized_train_set = train_set.map(_tokenize_function, batched=True)
    tokenized_train_set.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    sample_size = kwargs.get("sample_size", None)
    if sample_size:
        return tokenized_train_set.select(range(10))
    return tokenized_train_set
