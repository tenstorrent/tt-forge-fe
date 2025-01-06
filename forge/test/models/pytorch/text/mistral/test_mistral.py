# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, MistralConfig
import forge
from forge.transformers.pipeline import NLPPipelineWrapper
from test.models.pytorch.text.mistral.utils.model_utils import BaseModelWrapper
from test.models.utils import build_module_name, Framework
from forge.verify.verify import verify


variants = ["mistralai/Mistral-7B-v0.1"]


@pytest.mark.skip(reason="Tested as part of full model test run")
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.nightly
def test_mistral_decoder_layer(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="mistral", variant=variant, suffix="decoder")

    record_forge_property("module_name", module_name)

    model = AutoModelForCausalLM.from_pretrained(variant, device_map="auto")
    model.eval()

    framework_model = model.model.layers[0]

    # test should work for batch size 1 and seqlen <= 128
    # for larger seqlen, a problem with valid node placement can occur
    batch_size = 1
    hidden_dim = 4096
    seqlen = 128

    sample_inputs = torch.randn(batch_size, seqlen, hidden_dim)

    inputs = [sample_inputs]

    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    verify(inputs, framework_model, compiled_model)


variants = ["mistralai/Mistral-7B-v0.1"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_mistral(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="mistral", variant=variant)

    record_forge_property("module_name", module_name)

    configuration = MistralConfig()
    configuration.sliding_window = None
    configuration.use_cache = False
    configuration.return_dict = False

    framework_model = AutoModelForCausalLM.from_pretrained(variant, device_map="auto", config=configuration)
    tokenizer = AutoTokenizer.from_pretrained(variant)

    framework_model.eval()
    for param in framework_model.parameters():
        param.requires_grad = False

    # test should work for batch size 1 and seqlen <= 128
    # for larger seqlen, a DRAM allocation problem might occur (this model is already near maximum model size for single chip)
    prompt = "Of course, fancy writing doesn't just conceal ideas. It can also conceal the lack of them. That's why some people write that way, to conceal the fact that they have nothing to say. Whereas writing simply keeps you honest. If you say nothing simply, it will be obvious to everyone, including you. Simple writing also lasts better. People reading your stuff in the future will be in much the same position as people from other countries reading it today. The culture and the language will have changed. It's not vain to care about that, any more than it's vain for "
    sample_inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
    inputs = [sample_inputs]

    compiled_model = forge.compile(
        framework_model,
        inputs,
        module_name,
    )

    verify(inputs, framework_model, compiled_model)


variants = ["mistralai/Mistral-7B-v0.1"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.skip(reason="This test currently serves the same purpose as test_mistral")
def test_mistral_decode(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="mistral", variant=variant, suffix="decode")

    record_forge_property("module_name", module_name)

    configuration = MistralConfig()
    configuration.sliding_window = None
    configuration.use_cache = False
    configuration.return_dict = False

    pytorch_model = AutoModelForCausalLM.from_pretrained(variant, device_map="auto", config=configuration)
    tokenizer = AutoTokenizer.from_pretrained(variant)

    pytorch_model.eval()
    for param in pytorch_model.parameters():
        param.requires_grad = False

    tokenizer.pad_token = tokenizer.eos_token

    prompt = "Of course, fancy writing doesn't just conceal ideas. It can also conceal the lack of them. That's why some people write that way, to conceal the fact that they have nothing to say. Whereas writing simply keeps"
    inputs = tokenizer(prompt, return_tensors="pt")

    max_generated_tokens = 100

    generate_ids = pytorch_model.generate(inputs.input_ids, max_length=max_generated_tokens)
    generated_pt_text = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("Based on prompt:")
    print(f"{prompt}")
    print(f"\nPyTorch (sanity) generated:")
    pt_ans = generated_pt_text.split("\n\n")
    print(f"{pt_ans}")

    wrapper = NLPPipelineWrapper(
        pytorch_model,
        tokenizer,
        pytorch_model.__class__.__name__,
        use_cache=None,
        forward_fn=None,
        max_length=max_generated_tokens,
    )

    pytorch_model.prepare_inputs_for_generation = wrapper.prepare_inputs_for_generation

    # this generates sample text, to trigger model compilation, so it is not factored during latency measurement
    outputs = pytorch_model.generate(inputs["input_ids"][:, 0:1], do_sample=False, max_length=max_generated_tokens)
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    start = time.time()
    outputs = pytorch_model.generate(inputs["input_ids"], do_sample=False, max_length=max_generated_tokens)
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    end = time.time()

    num_generated_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
    print("TT generated:")
    print(output_text[0])
    print(f"Tokens / s: {num_generated_tokens / (end-start)}")


variants = ["mistralai/Mistral-7B-v0.1"]


@pytest.mark.nightly
@pytest.mark.skip(reason="under development")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_mistral_kv_cache(record_forge_property, variant, test_device):
    module_name = build_module_name(framework=Framework.PYTORCH, model="mistral", variant=variant, suffix="kv_cache")

    record_forge_property("module_name", module_name)

    configuration = MistralConfig()
    configuration.sliding_window = None
    configuration.use_cache = True
    configuration.return_dict = False

    max_new_tokens = 10

    # configuration for all ops that are not matmul
    forge.config.configure_mixed_precision(
        op_type="^((?!matmul).)*$", math_fidelity=MathFidelity.HiFi4, accumulate_df=DataFormat.Float16_b
    )

    # configuration for all matmul ops
    # when inputs to matmuls are Bfp8_b, the whole model can fit to single chip
    forge.config.configure_mixed_precision(
        op_type="matmul",
        math_fidelity=MathFidelity.HiFi4,
        input_df={0: [DataFormat.Bfp8_b, False], 1: [DataFormat.Bfp8_b, False]},
        accumulate_df=DataFormat.Float16_b,
    )

    model = AutoModelForCausalLM.from_pretrained(variant, device_map="auto", config=configuration)
    tokenizer = AutoTokenizer.from_pretrained(variant)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    tokenizer.pad_token = tokenizer.eos_token

    prompt = "Of course, fancy writing doesn't just conceal ideas. It can also conceal the lack of them. That's why some people write that way, to conceal the fact that they have nothing to say. Whereas writing simply keeps"

    inputs = tokenizer(prompt, return_tensors="pt")

    T = inputs["input_ids"].shape[-1]
    output_ids = inputs["input_ids"].clone()
    position_ids = torch.arange(T)
    inputs = tuple(inputs.values())
    inputs += (position_ids,)

    # perform prefill with torch model on cpu
    logits, past_key_values = model(*inputs)

    tt1 = forge.TTDevice(
        "tt1",
        devtype=test_device.devtype,
        arch=test_device.arch,
        module=PyTorchModule("mistral_model_base", BaseModelWrapper(model)),
    )

    next_token = sample(logits)
    output_ids = torch.cat([output_ids, next_token], axis=1)
    position_ids = torch.tensor([[T]])
    mask = torch.ones(1, T + 1)

    inputs = (
        next_token,
        mask,
        position_ids,
    )
    for i in range(configuration.num_hidden_layers):
        inputs += (past_key_values[i][0], past_key_values[i][1])

    # compile model before measuring perf
    output_q = forge.initialize_pipeline(
        training=False, sample_inputs=inputs, _sequential=True, _device_mode=DeviceMode.CompileAndRun
    )

    start_time = time.time()
    for i in range(max_new_tokens):

        position_ids = torch.tensor([[T]])
        mask = torch.ones(1, T + 1)
        if i > 0:  # for i = 0 we have already defined inputs
            inputs = (next_token, mask, position_ids, *past_key_values)

        tt1.push_to_inputs(inputs)
        forge.run_forward(input_count=1, _sequential=True)
        outputs = output_q.get()

        logits = outputs[0].value().to(dtype=torch.float)

        next_token = sample(logits)
        output_ids = torch.cat([output_ids, next_token], axis=1)
        past_key_values = [el.value() for el in outputs[1:]]
        T += 1

    duration = time.time() - start_time

    tokens_per_second = max_new_tokens / duration
    generated_text = tokenizer.decode(output_ids[0].numpy().tolist())
    print(generated_text)
    print(f"Tokens per second: {tokens_per_second}")
