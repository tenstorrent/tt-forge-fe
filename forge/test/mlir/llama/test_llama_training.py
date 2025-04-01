# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.utils.data import DataLoader

import forge
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
import forge.optimizers
from forge.verify.verify import verify, verify_backward
from test.mlir.llama.utils.utils import load_model, load_tokenized_data


@pytest.mark.parametrize("model_path", ["meta-llama/Llama-3.2-1B", "openlm-research/open_llama_3b"])
def test_llama_lora_fwd_pass(forge_property_recorder, model_path):
    if model_path == "openlm-research/open_llama_3b":
        pytest.skip(
            "TT_THROW: Out of Memory: Not enough space to allocate 110592000 B DRAM buffer across 12 banks, where each bank needs to store 9216000 B"
        )

    # Load Model and Tokenizer for LoRA training
    use_lora = True
    framework_model, tokenizer = load_model(model_path, use_lora=use_lora)

    # Need input seq divisible by 32 due to metal constraints TILE_WIDTH=32
    # Can be changed when https://github.com/tenstorrent/tt-metal/issues/17714 resolved
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 32

    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids

    # Compile the model for forward pass
    compiled_model = forge.compile(framework_model, input_ids, forge_property_handler=forge_property_recorder)

    verify([input_ids], framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize("model_path", ["meta-llama/Llama-3.2-1B", "openlm-research/open_llama_3b"])
def test_llama_lora_bwd_pass(forge_property_recorder, model_path):
    if model_path == "openlm-research/open_llama_3b":
        pytest.skip(
            "TT_THROW: Out of Memory: Not enough space to allocate 110592000 B DRAM buffer across 12 banks, where each bank needs to store 9216000 B"
        )

    # Load Model and Tokenizer for LoRA training
    use_lora = True
    framework_model, tokenizer = load_model(model_path, use_lora=use_lora)
    framework_model.train()

    # Need input seq divisible by 32 due to metal constraints TILE_WIDTH=32
    # Can be changed when https://github.com/tenstorrent/tt-metal/issues/17714 resolved
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 32

    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids

    # Create a batch of input_ids
    batch_size = 32
    input_ids = input_ids.repeat(batch_size, 1)

    # Compile the model for training
    compiled_model = forge.compile(
        framework_model, input_ids, training=True, forge_property_handler=forge_property_recorder
    )

    fw_out, co_out = verify(
        [input_ids], framework_model, compiled_model, forge_property_handler=forge_property_recorder
    )

    # Run bwd pass
    grad = torch.rand_like(fw_out[0])

    verify_backward(
        [input_ids],
        grad,
        fw_out[0],
        co_out[0],
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.99)),
    )


@pytest.mark.parametrize("model_path", ["meta-llama/Llama-3.2-1B", "openlm-research/open_llama_3b"])
@pytest.mark.parametrize("dataset_id", ["stanfordnlp/sst2"])
def test_llama_training(model_path, dataset_id):
    if model_path == "openlm-research/open_llama_3b":
        pytest.skip(
            "TT_THROW: Out of Memory: Not enough space to allocate 110592000 B DRAM buffer across 12 banks, where each bank needs to store 9216000 B"
        )

    # Setup hyperparameters
    num_epoch = 3
    max_length = 128
    batch_size = 4
    num_layers = 20

    # Load Model and Tokenizer
    use_lora = True
    framework_model, tokenizer = load_model(model_path, use_lora=use_lora, num_hidden_layers=num_layers)
    framework_model.train()

    # Need input seq divisible by 32 due to metal constraints TILE_WIDTH=32
    # Can be changed when https://github.com/tenstorrent/tt-metal/issues/17714 resolved
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_length

    tokenized_data = load_tokenized_data(dataset_id, tokenizer, max_length=max_length, sample_size=100)
    data_loader = DataLoader(tokenized_data, batch_size=batch_size, shuffle=False, drop_last=True)

    tt_optimizer = forge.optimizers.AdamW()

    # Compile the model for training
    sample_inputs = [torch.randint(0, framework_model.config.vocab_size, (batch_size, max_length))]
    compiled_model = forge.compile(framework_model, sample_inputs, optimizer=tt_optimizer, training=True)

    # Create a torch loss and leave on CPU
    # Can be changed when https://github.com/tenstorrent/tt-metal/issues/18997 resolved
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    losses = []
    for epoch in range(num_epoch):
        epoch_loss = 0

        for batch in data_loader:
            # Fwd pass
            input_ids = batch["input_ids"]
            logits = compiled_model(input_ids)[0]

            # Bwd pass
            # Create label tensor - set to -100 for prompt tokens (they shouldn't contribute to loss)
            prompt_length = len(input_ids)
            labels = batch["labels"]
            labels_for_loss = labels.clone()
            labels_for_loss[0, :prompt_length] = -100

            loss = loss_fn(logits.view(-1, framework_model.config.vocab_size), labels_for_loss.view(-1))
            epoch_loss += loss.item()

            loss.backward()
            compiled_model.backward()

            tt_optimizer.step()

        losses.append(epoch_loss / len(tokenized_data))

    for epoch, epoch_loss in enumerate(losses):
        print(f"Epoch: {epoch+1}, Loss: {epoch_loss}")
