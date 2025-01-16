# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Whisper Demo - Conditional Generation
# Example of ASR pipeline: https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/tests/pipelines/test_pipelines_automatic_speech_recognition.py#L695

import os

import pytest
import torch
from transformers import (
    AutoProcessor,
    LogitsProcessorList,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)

import forge
from forge.config import _get_global_compiler_config
from forge.forgeglobal import TILE_DIM
from forge.transformers.pipeline import pipeline as forge_pipeline

from test.models.pytorch.audio.whisper.utils.model import (
    Whisper_decoder,
    Whisper_encoder,
    generate_model_whisper_decoder_past_cache,
)
from test.models.utils import Framework, build_module_name
from test.utils import download_model

variants = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large",
]


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.skip(reason="Redundant")
def test_whisper_dec_past_cache(record_forge_property, test_device, variant):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="whisper", variant=variant, suffix="pipeline")

    # Record Forge Property
    record_forge_property("model_name", module_name)

    model, inputs, other = generate_model_whisper_decoder_past_cache(variant)
    compile_inputs = other["compile_inputs"]
    max_length = other["max_length"]
    tt0 = forge.TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch, module=model)

    output_q = forge.initialize_pipeline(
        training=False,
        sample_inputs=compile_inputs,
    )

    import time

    for _ in range(10):
        start = time.time()
        tokens_to_generate = 64 if test_device.devtype == BackendType.Silicon else 3
        for _ in range(tokens_to_generate):
            tt0.push_to_inputs(inputs)
            forge.run_generate(input_count=1, write_index=0)
            ans = output_q.get()

        end = time.time()
        print(
            f"Iteration 0: {tokens_to_generate} iterations took {end - start} seconds, speed: {(tokens_to_generate) / (end - start)} iters/sec"
        )
        if test_device.devtype != BackendType.Silicon:
            break


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
@pytest.mark.skip(reason="not supported yet")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_whisper_enc_dec(record_forge_property, test_device, variant):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="whisper", variant=variant, suffix="enc_dec")

    # Record Forge Property
    record_forge_property("model_name", module_name)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False  # Run full model on silicon
    compiler_cfg.input_queues_on_host = True
    compiler_cfg.compile_subgraphs = True
    compiler_cfg.enable_link_past_cache_ios = True
    compiler_cfg.backend_opt_level = 4
    compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b
    os.environ["FORGE_FORCE_SEQUENTIAL"] = "1"

    run_encoder_on_tt = ("tiny" in variant) or ("base" in variant) or ("small" in variant) or ("medium" in variant)

    pad_model = True
    # forge.set_configuration_options(performance_trace=forge.PerfTraceLevel.VERBOSE)
    processor = download_model(AutoProcessor.from_pretrained, variant)
    config = WhisperConfig.from_pretrained(variant)
    config.return_dict = False

    # config.encoder_layers = 2
    # config.decoder_layers = 2
    if pad_model:
        config.max_source_positions = 1536

    max_length = config.max_length
    model = download_model(
        WhisperForConditionalGeneration.from_pretrained,
        variant,
        ignore_mismatched_sizes=True,
        config=config,
    )
    if pad_model:
        unpadded_model = WhisperForConditionalGeneration.from_pretrained(variant)
        padded_param = torch.nn.functional.pad(unpadded_model.model.encoder.embed_positions.weight.data, (0, 0, 0, 36))
        model.model.encoder.embed_positions.weight.data = padded_param

    feature_extractor = download_model(WhisperFeatureExtractor.from_pretrained, variant)
    tokenizer = WhisperTokenizer.from_pretrained(variant)
    encoder_module = forge.PyTorchModule("Whisper_encoder", Whisper_encoder(model))
    decoder_module_cross_attention = forge.PyTorchModule("Whisper_decoder_with_ca", Whisper_decoder(model))
    decoder_module_no_cross_attention = forge.PyTorchModule("Whisper_decoder_no_ca", Whisper_decoder(model))

    # ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    smaller_dataset = []
    if True:  # test_device.devtype != BackendType.Silicon:
        for i in range(1):
            sample = torch.load(f"forge/test/models/files/samples/audio/1272-128104-000{i}.pt")
            smaller_dataset.append(sample)
            # smaller_dataset.append(ds[i])
        ds = smaller_dataset
    inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    # sample = torch.load("forge/test/models/files/samples/audio/1272-128104-0000.pt")
    # sample_audio = sample["audio"]["array"]

    # inputs = processor(sample_audio, return_tensors="pt")
    if pad_model:
        input_features = torch.nn.functional.pad(inputs.input_features, (0, 72, 0, 0))
    else:
        input_features = inputs.input_features

    encoder_last_hidden_state_shape = (1, config.max_source_positions, config.d_model)
    encoder_last_hidden_state = torch.zeros(encoder_last_hidden_state_shape)

    logits_processor = model._get_logits_processor(
        model.generation_config, TILE_DIM, input_features, None, LogitsProcessorList()
    )
    decoder_attention_mask = torch.zeros((1, max_length))
    decoder_input_ids = torch.ones((1, TILE_DIM), dtype=torch.int) * tokenizer.pad_token_id
    first_current_index = max_length - TILE_DIM
    position_embeds = torch.zeros((TILE_DIM, config.d_model))
    enc_past_cache_self_shape = (
        1,
        config.decoder_attention_heads,
        max_length - TILE_DIM,
        config.d_model // config.decoder_attention_heads,
    )
    enc_past_cache_cross_shape = (1, 1, 1, 1)

    decoder_with_ca_inputs = [decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, position_embeds]
    for _ in range(config.decoder_layers):
        decoder_with_ca_inputs += [
            torch.zeros(enc_past_cache_self_shape),
            torch.zeros(enc_past_cache_self_shape),
            torch.zeros(enc_past_cache_cross_shape),
            torch.zeros(enc_past_cache_cross_shape),
        ]

    dec = Whisper_decoder(model)
    dec(*decoder_with_ca_inputs)
    enc_past_cache_cross_shape = (
        1,
        config.decoder_attention_heads,
        config.max_source_positions,
        config.d_model // config.decoder_attention_heads,
    )
    decoder_no_ca_inputs = [decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, position_embeds]
    for _ in range(config.decoder_layers):
        decoder_no_ca_inputs += [
            torch.zeros(enc_past_cache_self_shape),
            torch.zeros(enc_past_cache_self_shape),
            torch.zeros(enc_past_cache_cross_shape),
            torch.zeros(enc_past_cache_cross_shape),
        ]

    if run_encoder_on_tt:
        tt0 = forge.TTDevice(
            "tt0",
            devtype=test_device.devtype,
            arch=test_device.arch,
            module=[encoder_module, decoder_module_cross_attention, decoder_module_no_cross_attention],
        )

        output_q = forge.initialize_pipeline(
            training=False,
            sample_inputs=(
                (input_features,),
                (decoder_with_ca_inputs),
                (decoder_no_ca_inputs),
            ),
        )
    else:
        tt0 = forge.TTDevice(
            "tt0",
            devtype=test_device.devtype,
            arch=test_device.arch,
            module=[decoder_module_cross_attention, decoder_module_no_cross_attention],
        )

        output_q = forge.initialize_pipeline(
            training=False,
            sample_inputs=(
                (decoder_with_ca_inputs),
                (decoder_no_ca_inputs),
            ),
        )

    import time

    for datum in ds:
        inputs = feature_extractor(datum["audio"]["array"], return_tensors="pt")

        if pad_model:
            input_features = torch.nn.functional.pad(inputs.input_features, (0, 72, 0, 0))
        else:
            input_features = inputs.input_features
        decoder_attention_mask = torch.zeros((1, max_length))
        decoder_input_ids[0, 0] = tokenizer.encode("<|startoftranscript|>")[0]
        decoder_attention_mask[0, first_current_index] = 1
        current_token_index = 0

        prefix_tokens = processor.get_decoder_prompt_ids(language="english", task="transcribe")
        for idx, token in prefix_tokens:
            decoder_input_ids[0, idx] = token
            decoder_attention_mask[0, first_current_index + idx] = 1
            current_token_index = idx

        # encoder hangs for some variants, for now run on cpu
        encoder_last_hidden_state = model.model.encoder(input_features)[0].detach()
        start = time.time()
        first_active_subgraph = 0
        if run_encoder_on_tt:
            tt0.set_active_subgraph(0)
            tt0.push_to_inputs((input_features,))
            forge.run_forward()
            ans = output_q.get()
            encoder_last_hidden_state = ans[0].value().detach()
            first_active_subgraph = 1
        generated_tokens = []
        encoder_last_hidden_state_consumed = False
        position_ids = torch.arange(32, dtype=torch.long)
        position_embeds = model.model.decoder.embed_positions.weight[position_ids]
        tokens_to_generate = 64 if test_device.devtype == BackendType.Silicon else 3
        for _ in range(tokens_to_generate):
            if not encoder_last_hidden_state_consumed:
                start_1 = time.time()
                encoder_last_hidden_state_consumed = True
                tt0.set_active_subgraph(first_active_subgraph)
                generate_inputs = (
                    decoder_input_ids,
                    decoder_attention_mask,
                    encoder_last_hidden_state,
                    position_embeds,
                )
                tt0.push_to_inputs(generate_inputs)
                forge.run_generate(input_count=1, write_index=current_token_index // TILE_DIM)
                ans = output_q.get()
                tt0.set_active_subgraph(first_active_subgraph + 1)
                start_2 = time.time()
            else:
                generate_inputs = (decoder_input_ids, decoder_attention_mask, position_embeds)
                tt0.push_to_inputs(generate_inputs)
                forge.run_generate(input_count=1, write_index=current_token_index // TILE_DIM)
                ans = output_q.get()

            lm_head_out = ans[0].value().detach()
            scores = logits_processor(
                decoder_input_ids[:, :current_token_index], lm_head_out[:, current_token_index % TILE_DIM]
            )
            next_token = torch.argmax(scores, dim=-1).item()
            generated_tokens.append(next_token)

            current_token_index += 1
            if current_token_index % TILE_DIM == 0 and current_token_index != max_length:
                position_ids = position_ids + TILE_DIM
                position_embeds = model.model.decoder.embed_positions.weight[position_ids]
                decoder_attention_mask[0, :current_token_index] = 1
                decoder_attention_mask[0, first_current_index:] = 0
                decoder_input_ids[0, :] = tokenizer.pad_token_id

            decoder_input_ids[0, current_token_index % TILE_DIM] = next_token
            decoder_attention_mask[0, first_current_index + (current_token_index % TILE_DIM)] = 1
        end = time.time()
        print(
            f"{tokens_to_generate} iterations took {end - start} seconds, speed: {(tokens_to_generate) / (end - start)} iters/sec"
        )
        print(
            f"{(tokens_to_generate)} iterations took {end - start_1} seconds, speed: {(tokens_to_generate) / (end - start_1)} iters/sec"
        )
        print(
            f"{(tokens_to_generate - 1)} iterations took {end - start_2} seconds, speed: {(tokens_to_generate - 1) / (end - start_2)} iters/sec"
        )
        print(f"Encoder took: {start_1 - start} seconds")
        print(f"Decoder with CA took: {start_2 - start_1} seconds")
        print(f"Decoder without CA took: {(end - start_2) / (tokens_to_generate - 2)} seconds")
        print(f"generated tokens: {tokenizer.decode(generated_tokens)}")


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.skip(reason="Redundant")
def test_whisper_enc_dec_pipeline(record_forge_property, test_device, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="whisper", variant=variant, suffix="enc_dec_pipeline"
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False  # Run full model on silicon
    compiler_cfg.input_queues_on_host = True
    compiler_cfg.compile_subgraphs = True
    compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b
    compiler_cfg.enable_link_past_cache_ios = True
    os.environ["FORGE_FORCE_SEQUENTIAL"] = "1"

    # forge.set_configuration_options(performance_trace=forge.PerfTraceLevel.VERBOSE)
    feature_extractor = download_model(WhisperFeatureExtractor.from_pretrained, variant)
    processor = download_model(AutoProcessor.from_pretrained, variant)
    config = WhisperConfig.from_pretrained(variant)
    max_length = config.max_length
    # max_length = 32
    model = download_model(
        WhisperForConditionalGeneration.from_pretrained,
        variant,
        return_dict=False,
    )
    tokenizer = WhisperTokenizer.from_pretrained(variant)
    encoder_module = forge.PyTorchModule("Whisper_encoder", Whisper_encoder(model))
    decoder_module_cross_attention = forge.PyTorchModule("Whisper_decoder_with_ca", Whisper_decoder(model))
    decoder_module_no_cross_attention = forge.PyTorchModule("Whisper_decoder_no_ca", Whisper_decoder(model))

    sample = torch.load("forge/test/models/files/samples/audio/1272-128104-0000.pt")
    sample_audio = sample["audio"]["array"]

    inputs = processor(sample_audio, return_tensors="pt")
    input_features = inputs.input_features

    encoder_last_hidden_state_shape = (1, config.max_source_positions, config.d_model)
    encoder_last_hidden_state = torch.zeros(encoder_last_hidden_state_shape)

    model.generate(input_features)
    logits_processor = model._get_logits_processor(
        model.generation_config, TILE_DIM, input_features, None, LogitsProcessorList()
    )
    sequence_length = 1500
    decoder_attention_mask = torch.zeros((1, max_length))
    decoder_input_ids = torch.ones((1, TILE_DIM), dtype=torch.int) * tokenizer.pad_token_id
    first_current_index = max_length - TILE_DIM
    position_embeds = torch.zeros((TILE_DIM, config.d_model))
    enc_past_cache_self_shape = (
        1,
        config.decoder_attention_heads,
        max_length - TILE_DIM,
        config.d_model // config.decoder_attention_heads,
    )
    enc_past_cache_cross_shape = (1, 1, 1, 1)

    decoder_with_ca_inputs = [decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, position_embeds]
    for _ in range(config.decoder_layers):
        decoder_with_ca_inputs += [
            torch.zeros(enc_past_cache_self_shape),
            torch.zeros(enc_past_cache_self_shape),
            torch.zeros(enc_past_cache_cross_shape),
            torch.zeros(enc_past_cache_cross_shape),
        ]
    enc_past_cache_cross_shape = (
        1,
        config.decoder_attention_heads,
        sequence_length,
        config.d_model // config.decoder_attention_heads,
    )
    decoder_no_ca_inputs = [decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, position_embeds]
    for _ in range(config.decoder_layers):
        decoder_no_ca_inputs += [
            torch.zeros(enc_past_cache_self_shape),
            torch.zeros(enc_past_cache_self_shape),
            torch.zeros(enc_past_cache_cross_shape),
            torch.zeros(enc_past_cache_cross_shape),
        ]

    tt0 = forge.TTDevice(
        "tt0",
        devtype=test_device.devtype,
        arch=test_device.arch,
        module=[decoder_module_cross_attention, decoder_module_no_cross_attention],
    )
    # module=[encoder_module, decoder_module_cross_attention, decoder_module_no_cross_attention])

    output_q = forge.initialize_pipeline(
        training=False,
        sample_inputs=(
            # (input_features,),
            (decoder_with_ca_inputs),
            (decoder_no_ca_inputs),
        ),
    )

    current_token_index = 0

    # encoder hangs, for now run on cpu
    generated_tokens = []
    encoder_last_hidden_state_consumed = False
    position_embeds = model.model.decoder.embed_positions.weight[:TILE_DIM]

    def wrap_generate(inputs):
        nonlocal current_token_index, encoder_last_hidden_state_consumed, generated_tokens
        encoder_last_hidden_state = inputs[1]
        generated_tokens.append(inputs[0][:, current_token_index % TILE_DIM].item())
        print(f"generated tokens: {tokenizer.decode(generated_tokens)}")
        decoder_input_ids[:, current_token_index % TILE_DIM] = inputs[0][:, current_token_index]
        decoder_attention_mask[0, first_current_index + (current_token_index % TILE_DIM)] = 1
        generate_inputs = (decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, position_embeds)
        tt0.set_active_subgraph(0)
        tt0.push_to_inputs(generate_inputs)
        forge.run_generate(input_count=1, write_index=0)
        ans = output_q.get()
        lm_head_out = ans[0].value().detach()
        lm_head_out = lm_head_out[:, : (current_token_index % TILE_DIM) + 1, :]

        current_token_index += 1
        if current_token_index % TILE_DIM == 0:
            decoder_attention_mask[0, :current_token_index] = 1
            decoder_attention_mask[0, first_current_index:] = 1
            decoder_input_ids[0, :] = tokenizer.pad_token_id
        return lm_head_out

    asr_pipeline = forge_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        forward_fn=wrap_generate,
    )

    tt_out = asr_pipeline(sample_audio)
    print("TT pipeline:", tt_out["text"])
