# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import torch
import forge
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config


@pytest.mark.nightly
def test_gpt2_text_gen(test_device):
    # Load tokenizer and model from HuggingFace
    config = GPT2Config.from_pretrained("gpt2")
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GPT2Config(**config_dict)
    model = download_model(GPT2LMHeadModel.from_pretrained, "gpt2", config=config)

    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # Wrapper to get around past key values
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids, None, attention_mask)

    input_ids = torch.cat(
        [torch.randint(1, model.config.vocab_size, (1, 255)), torch.zeros(1, 1, dtype=torch.int64)], dim=-1
    ).to(torch.int64)
    attn_mask = torch.ones(1, 256)
    inputs = [input_ids, attn_mask]
    compiled_model = forge.compile(Wrapper(model), sample_inputs=inputs, module_name="pt_gpt2_generation")


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, *kv):
        num_past_key_values = len(kv)
        past_key_values = None if num_past_key_values == 0 else []
        for i in range(num_past_key_values // 2):
            past_key_values.append((kv[2 * i], kv[2 * i + 1]))

        return self.model(input_ids, past_key_values, attention_mask)


@pytest.mark.nightly
@pytest.mark.skip(reason="not supported yet")
def test_gpt2_past_cache(test_device):
    os.environ["GOLDEN_WORMHOLE_B0"] = "1"
    os.environ["FORGE_DEVMODE"] = "1"
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_subgraphs = True
    compiler_cfg.enable_tvm_cpu_fallback = False
    compiler_cfg.enable_auto_fusing = False

    model = GPT2LMHeadModel.from_pretrained("gpt2", return_dict=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config = GPT2Config.from_pretrained("gpt2")
    config_dict = config.to_dict()
    config_dict["n_layer"] = 2
    config_dict["return_dict"] = False
    config = GPT2Config(**config_dict)
    model = download_model(GPT2LMHeadModel.from_pretrained, "gpt2", config=config)

    tokenizer.pad_token = tokenizer.eos_token

    run_length = 480
    prefix_text = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains."
    inputs = tokenizer(prefix_text, max_length=run_length, pad_to_max_length=True, truncation=True, return_tensors="pt")
    inputs = [inputs["input_ids"].int(), inputs["attention_mask"].float()]

    tt0 = forge.TTDevice("tt0")
    tt0.place_module(module=forge.PyTorchModule("gpt2", Wrapper(model)))
    tt0.push_to_inputs(inputs)
    output_q = forge.initialize_pipeline(
        training=False,
    )
    forge.run_forward()
    res = output_q.get()

    tt0.remove_modules()
    tt0.place_module(module=forge.PyTorchModule("gpt2", Wrapper(model)))

    inputs.extend([res[1].value(), res[2].value(), res[3].value(), res[4].value()])
    inputs[1] = torch.cat((inputs[1], (torch.zeros((1, 32)))), 1)
    inputs[0] = inputs[0][:, :32]
    tt0.push_to_inputs(inputs)
    output_q = forge.initialize_pipeline(
        training=False,
    )
    forge.run_forward()
    breakpoint()
