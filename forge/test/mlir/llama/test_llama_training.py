# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import forge
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify, verify_backward
from test.mlir.llama.utils.utils import load_model, load_tokenized_data, batchify
from torch.utils.data import DataLoader
from forge.config import CompilerConfig
from forge._C import DataFormat


@pytest.mark.parametrize("model_path", ["meta-llama/Llama-3.2-1B", "openlm-research/open_llama_3b"])
@pytest.mark.push
def test_llama_lora_fwd_pass(model_path):
    if model_path == "openlm-research/open_llama_3b":
        pytest.skip("Insufficient host DRAM to run this model")

    # Load Model and Tokenizer for LoRA training
    framework_model, tokenizer = load_model(model_path, use_lora=True)

    # Need input seq divisible by 32 due to metal constraints TILE_WIDTH=32
    # Can be changed when https://github.com/tenstorrent/tt-metal/issues/17714 resolved
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 32

    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids

    # Compile the model for forward pass
    compiled_model = forge.compile(framework_model, input_ids)

    verify([input_ids], framework_model, compiled_model)


@pytest.mark.parametrize("model_path", ["meta-llama/Llama-3.2-1B", "openlm-research/open_llama_3b"])
@pytest.mark.push
def test_llama_lora_bwd_pass(model_path):
    # Load Model and Tokenizer for LoRA training
    # NOTE: Using only 1 hidden layer for CI testing purposes.
    # Full models fails on 0.99 PCC on some layers, but passes above 0.90.
    # Also, not enough DRAM memory to run full open llama 3B full model.
    framework_model, tokenizer = load_model(model_path, use_lora=True, num_hidden_layers=1)
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
    compiled_model = forge.compile(framework_model, input_ids, training=True)

    fw_out, co_out = verify([input_ids], framework_model, compiled_model)

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
@pytest.mark.push
def test_llama_lora_bfloat16(forge_property_recorder, model_path):
    # Load Model and Tokenizer for LoRA training
    # NOTE: Using only 1 hidden layer for CI testing purposes.
    # Full models fails on 0.99 PCC on some layers, but passes above 0.90.
    framework_model, tokenizer = load_model(model_path, use_lora=True, num_hidden_layers=1)
    framework_model.to(torch.bfloat16)
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

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Compile the model for training
    compiled_model = forge.compile(
        framework_model,
        input_ids,
        training=True,
        compiler_cfg=compiler_cfg,
    )

    fw_out, co_out = verify([input_ids], framework_model, compiled_model)

    # Run bwd pass
    grad = torch.rand_like(fw_out[0]).to(torch.bfloat16)

    verify_backward(
        [input_ids],
        grad,
        fw_out[0],
        co_out[0],
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.99)),
    )


@pytest.mark.parametrize("model_path", ["meta-llama/Llama-3.2-1B"])
@pytest.mark.parametrize("dataset_id", ["stanfordnlp/sst2"])
def test_llama_training(model_path, dataset_id):
    # Setup hyperparameters
    num_epoch = 3
    max_length = 64
    batch_size = 16
    num_layers = 16

    # Load Model and Tokenizer
    use_lora = True
    framework_model, tokenizer = load_model(model_path, use_lora=use_lora, num_hidden_layers=num_layers)
    framework_model.to(torch.bfloat16)
    framework_model.train()

    # Need input seq divisible by 32 due to metal constraints TILE_WIDTH=32
    # Can be changed when https://github.com/tenstorrent/tt-metal/issues/17714 resolved
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_length

    tokenized_data = load_tokenized_data(dataset_id, tokenizer, max_length=max_length, sample_size=100)
    data_loader = DataLoader(tokenized_data, batch_size=batch_size, shuffle=False, drop_last=True)
    # data_loader = batchify(tokenized_data, batch_size=batch_size)

    torch_optimizer = torch.optim.AdamW(framework_model.parameters())
    # tt_optimizer = forge.optimizers.AdamW()

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Compile the model for training
    sample_inputs = [torch.randint(0, framework_model.config.vocab_size, (batch_size, max_length))]
    compiled_model = forge.compile(framework_model, sample_inputs, optimizer=torch_optimizer, training=True, compiler_cfg=compiler_cfg)

    # Create a torch loss and leave on CPU
    # Can be changed when https://github.com/tenstorrent/tt-metal/issues/18997 resolved
    loss_fn = torch.nn.CrossEntropyLoss()

    losses = []
    for epoch in range(num_epoch):
        epoch_loss = 0

        for idx, batch in enumerate(data_loader):
            torch_optimizer.zero_grad()

            # Fwd pass
            input_ids = batch["input_ids"]
            # input_ids = torch.stack(batch["input_ids"])
            logits = compiled_model(input_ids)[0]

            predicted_ids = torch.argmax(logits, dim=-1)
            with open("llama_training_predictions.txt", "a") as f:
                f.write("##### Input ids and Predicted text #####\n")
                f.write(f"Epoch {epoch}, Batch {idx}\n")
                f.write(tokenizer.decode(input_ids[0], skip_special_tokens=True) + "\n")
                f.write(tokenizer.decode(predicted_ids[0], skip_special_tokens=True) + "\n")
                f.write("\n")

            # Bwd pass
            expected_output = batch["labels"]
            # expected_output = torch.stack(batch["labels"])
            loss = loss_fn(logits.view(-1, framework_model.config.vocab_size), expected_output.view(-1))
            epoch_loss += loss.item()

            loss.backward()
            compiled_model.backward()

            # tt_optimizer.step()
            torch_optimizer.step()

        losses.append(epoch_loss / len(tokenized_data))

    for epoch, epoch_loss in enumerate(losses):
        print(f"Epoch: {epoch+1}, Loss: {epoch_loss}")
