# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GemmaConfig,
    GemmaForCausalLM,
)

import forge
from forge.transformers.pipeline import pipeline as forge_pipeline
from forge.verify.verify import verify

from test.models.utils import Framework, build_module_name
from test.utils import download_model


def cpu_sanity_run_0():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

    input_text = "Write me a poem about Machine Learning."
    input_ids = tokenizer(input_text, return_tensors="pt")

    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))


def cpu_sanity_run_1():
    model = GemmaForCausalLM.from_pretrained("google/gemma-2b")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

    prompt = "What is your favorite city?"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
        0
    ]

    print(generated_text)


variants = [
    "google/gemma-2b",
]


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
@pytest.mark.skip(reason="Tested as part of full model test run")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b_rotary_embedding(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="gemma", variant=variant, suffix="rotary_embedding"
    )

    # Record Forge Property
    record_forge_property("module_name", module_name)

    # Random see for reproducibility
    torch.manual_seed(42)

    # Load model
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model.layers[0].self_attn.rotary_emb

        def forward(self, x, pos_ids):
            cos, sin = self.model(x, pos_ids)

            return cos, sin

    config = download_model(GemmaConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GemmaConfig(**config_dict)
    framework_model = download_model(GemmaForCausalLM.from_pretrained, variant, config=config)
    framework_model = Wrapper(framework_model)

    # Define inputs
    x = torch.rand((1, 1, 7, 256)).to(torch.float32)
    pos_ids = torch.arange(7).unsqueeze(0).to(torch.float32)

    inputs = [x, pos_ids]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
@pytest.mark.skip(reason="Tested as part of full model test run")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b_rms_norm(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="gemma", variant=variant, suffix="rms_norm")

    # Record Forge Property
    record_forge_property("module_name", module_name)

    # Random see for reproducibility
    torch.manual_seed(42)

    # Load model
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model.layers[0].input_layernorm

        def forward(self, x):
            out = self.model(x)

            return out

    config = download_model(GemmaConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GemmaConfig(**config_dict)
    framework_model = download_model(GemmaForCausalLM.from_pretrained, variant, config=config)
    framework_model = Wrapper(framework_model)

    # Define inputs
    x = torch.rand((1, 7, 2048)).to(torch.float32)

    inputs = [x]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
@pytest.mark.skip(reason="Tested as part of full model test run")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b_attention(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="gemma", variant=variant, suffix="attention")

    # Record Forge Property
    record_forge_property("module_name", module_name)

    # Random see for reproducibility
    torch.manual_seed(42)

    # Load model
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model.layers[0].self_attn

        def forward(self, hidden_states, attn_mask, pos_ids):
            attn_output, attn_weights, past_key_value = self.model(hidden_states, attn_mask, pos_ids)

            return attn_output

    config = download_model(GemmaConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GemmaConfig(**config_dict)
    framework_model = download_model(GemmaForCausalLM.from_pretrained, variant, config=config)
    framework_model = Wrapper(framework_model)

    # Define inputs
    hidden_states = torch.rand((1, 7, 2048)).to(torch.float32)
    attn_mask = torch.ones((1, 1, 7, 7)).to(torch.float32)
    pos_ids = torch.arange(7).unsqueeze(0).to(torch.float32)

    inputs = [hidden_states, attn_mask, pos_ids]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
@pytest.mark.skip(reason="Tested as part of full model test run")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b_mlp(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="gemma", variant=variant, suffix="mlp")

    # Record Forge Property
    record_forge_property("module_name", module_name)

    # Random see for reproducibility
    torch.manual_seed(42)

    # Load model
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model.layers[0].mlp

        def forward(self, hidden_states):
            out = self.model(hidden_states)

            return out

    config = download_model(GemmaConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GemmaConfig(**config_dict)
    framework_model = download_model(GemmaForCausalLM.from_pretrained, variant, config=config)
    framework_model = Wrapper(framework_model)

    # Define inputs
    x = torch.rand((1, 7, 2048)).to(torch.float32)
    inputs = [x]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
@pytest.mark.skip(reason="Tested as part of full model test run")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b_single_decoder(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="gemma", variant=variant, suffix="single_decoder"
    )

    # Record Forge Property
    record_forge_property("module_name", module_name)

    # Random see for reproducibility
    torch.manual_seed(42)

    # Load model
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model.layers[0]

        def forward(self, hidden_states, attn_mask, pos_ids):
            out = self.model(hidden_states, attn_mask, pos_ids)

            return out

    config = download_model(GemmaConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GemmaConfig(**config_dict)
    framework_model = download_model(GemmaForCausalLM.from_pretrained, variant, config=config)
    framework_model = Wrapper(framework_model)

    # Define inputs
    hidden_states = torch.rand((1, 7, 2048)).to(torch.float32)
    attn_mask = torch.ones((1, 1, 7, 7)).to(torch.float32)
    pos_ids = torch.arange(7).unsqueeze(0).to(torch.float32)

    inputs = [hidden_states, attn_mask, pos_ids]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="gemma", variant=variant)

    # Record Forge Property
    record_forge_property("module_name", module_name)

    # Random see for reproducibility
    torch.manual_seed(42)

    config = download_model(GemmaConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GemmaConfig(**config_dict)
    framework_model = download_model(GemmaForCausalLM.from_pretrained, variant, config=config)

    # Load tokenizer
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token

    # Sample input
    prompt = "What is your favorite city?"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Sanity run
    generate_ids = framework_model.generate(inputs.input_ids, max_length=30)
    generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
        0
    ]

    print(f"Sanity run generated text: {generated_text}")

    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    inputs = [input_ids, attn_mask]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
@pytest.mark.skip(reason="Not supported yet")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b_gen(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="gemma", variant=variant, suffix="gen")

    # Record Forge Property
    record_forge_property("module_name", module_name)

    # Random seed for reproducibility
    torch.manual_seed(42)

    config = download_model(GemmaConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False

    config = GemmaConfig(**config_dict)
    pytorch_model = download_model(GemmaForCausalLM.from_pretrained, variant, config=config)

    # Load tokenizer
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token

    # Sample input
    prompt = "What is your favorite city?"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Sanity run
    generate_ids = pytorch_model.generate(inputs.input_ids, max_length=30)
    generated_pt_text = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("Based on prompt:")
    print(f"{prompt}")
    print(f"\nPyTorch (sanity) generated:")
    pt_ans = generated_pt_text.split("\n\n")[1]
    print(f"{pt_ans}")

    # Initialize and Run text2text generator on Tenstorrent device
    text2text_generator = forge_pipeline(
        "text2text-generation",
        model=pytorch_model,
        tokenizer=tokenizer,
        forge_max_length=32,
    )
    generated_tt_text = text2text_generator(
        prompt,
        max_length=32,
        num_beams=1,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
    )

    print("Based on prompt:")
    print(f"{prompt}")
    print(f"\nTT generated:")
    for sequence in generated_tt_text:
        tt_ans = sequence["generated_text"][len(prompt) :]
        print(f"{tt_ans}")


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
@pytest.mark.skip(reason="Not supported yet")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b_1x1_gen(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="gemma", variant=variant, suffix="gen_1x1")

    # Record Forge Property
    record_forge_property("module_name", module_name)

    # Random seed for reproducibility
    torch.manual_seed(42)

    config = download_model(GemmaConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False

    config = GemmaConfig(**config_dict)
    pytorch_model = download_model(GemmaForCausalLM.from_pretrained, variant, config=config)

    # Load tokenizer
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token

    # Sample input
    prompt = "What is your favorite city?"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Sanity run
    generate_ids = pytorch_model.generate(inputs.input_ids, max_length=30)
    generated_pt_text = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("Based on prompt:")
    print(f"{prompt}")
    print(f"\nPyTorch (sanity) generated:")
    pt_ans = generated_pt_text.split("\n\n")[1]
    print(f"{pt_ans}")

    # Initialize and Run text2text generator on Tenstorrent device
    text2text_generator = forge_pipeline(
        "text2text-generation",
        model=pytorch_model,
        tokenizer=tokenizer,
        forge_max_length=32,
    )
    generated_tt_text = text2text_generator(
        prompt,
        max_length=32,
        num_beams=1,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
    )

    print("Based on prompt:")
    print(f"{prompt}")
    print(f"\nTT generated:")
    for sequence in generated_tt_text:
        tt_ans = sequence["generated_text"][len(prompt) :]
        print(f"{tt_ans}")
