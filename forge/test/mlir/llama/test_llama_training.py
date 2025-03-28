# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import forge.optimizers
import pytest
import torch
from torch.utils.data import DataLoader

import forge

# from forge.op.loss import CrossEntropyLoss
# from forge.tensor import to_forge_tensors
from forge.verify.verify import verify
from test.mlir.llama.utils.utils import load_model, load_tokenized_data
import time


@pytest.mark.parametrize("model_path", ["meta-llama/Llama-3.2-1B", "openlm-research/open_llama_3b"])
@pytest.mark.parametrize("use_lora", [False])
def test_llama_fwd_pass(model_path, use_lora):
    if model_path == "openlm-research/open_llama_3b":
        pytest.skip(
            "TT_THROW: Out of Memory: Not enough space to allocate 110592000 B DRAM buffer across 12 banks, where each bank needs to store 9216000 B"
        )

    # Load Model and Tokenizer
    framework_model, tokenizer = load_model(model_path, use_lora=use_lora)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 32

    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids

    # Compile the model for forward pass
    compiled_model = forge.compile(framework_model, input_ids)

    verify([input_ids], framework_model, compiled_model)


@pytest.mark.parametrize("model_path", ["meta-llama/Llama-3.2-1B"])
@pytest.mark.parametrize("use_lora", [True])
def test_llama_bwd_pass(model_path, use_lora):
    # Load Model and Tokenizer
    framework_model, tokenizer = load_model(model_path, use_lora=use_lora)
    framework_model.train()

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 32

    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids

    # tt_optimizer = forge.optimizers.SGD(learning_rate=0.1)
    # framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=0.1)
    framework_optimizer = torch.optim.Adam(framework_model.parameters())

    # Compile the model for trainingc
    compiled_model = forge.compile(framework_model, input_ids, optimizer=framework_optimizer, training=True)

    logits = compiled_model(input_ids)[0]
    logits = logits.squeeze()
    labels = torch.nn.functional.one_hot(input_ids.squeeze(), num_classes=tokenizer.vocab_size).float()

    # loss_inputs = [torch.rand(32, 32000).requires_grad_(True), torch.rand(32, 32000)]
    # loss_inputs = to_forge_tensors(loss_inputs)
    # tt_loss = forge.compile(CrossEntropyLoss(name="cross_entropy_loss"), sample_inputs=loss_inputs, training=True, attach_to=compiled_model)

    # Create a torch loss and leave on CPU
    loss_fn = torch.nn.CrossEntropyLoss()

    loss = loss_fn(logits, labels)

    loss.backward()
    compiled_model.backward()

    framework_optimizer.step()
    framework_optimizer.zero_grad()


@pytest.mark.parametrize("model_path", ["meta-llama/Llama-3.2-1B", "openlm-research/open_llama_3b"])
@pytest.mark.parametrize("dataset_id", ["stanfordnlp/sst2"])
@pytest.mark.parametrize("use_lora", [True])
def test_llama_training(model_path, dataset_id, use_lora):
    if model_path == "meta-llama/Llama-3.2-1B":
        pytest.skip("NotImplementedError: repeat_interleave")

    num_epoch = 3
    max_length = 128
    batch_size = 4
    num_layers = 15

    # Load Model and Tokenizer
    framework_model, tokenizer = load_model(model_path, use_lora=use_lora, num_hidden_layers=num_layers)
    framework_model.train()

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_length

    tokenized_data = load_tokenized_data(dataset_id, tokenizer, max_length=max_length, sample_size=10)
    data_loader = DataLoader(tokenized_data, batch_size=batch_size, shuffle=False, drop_last=True)

    tt_optimizer = forge.optimizers.AdamW()

    sample_inputs = [torch.randint(0, tokenizer.vocab_size, (batch_size, max_length))]
    compiled_model = forge.compile(framework_model, sample_inputs, optimizer=tt_optimizer, training=True)

    # Compile the model for training
    losses = []
    total_start_time = time.time()
    breakpoint()
    for epoch in range(num_epoch):
        epoch_loss = 0

        for batch in data_loader:
            input_ids = batch["input_ids"]

            logits = compiled_model(input_ids)[0]
            logits = logits.view(-1, tokenizer.vocab_size)

            labels = input_ids.view(-1)

            # Create a torch loss and leave on CPU
            loss_fn = torch.nn.CrossEntropyLoss()

            loss = loss_fn(logits, labels)
            epoch_loss += loss.item()

            loss.backward()
            compiled_model.backward()

        losses.append(epoch_loss / len(tokenized_data))

        tt_optimizer.step()

    total_end_time = time.time()
    print(f"Total run time: {total_end_time - total_start_time:.2f} seconds")

    for epoch, epoch_loss in enumerate(losses):
        print(f"Epoch: {epoch+1}, Loss: {epoch_loss}")
