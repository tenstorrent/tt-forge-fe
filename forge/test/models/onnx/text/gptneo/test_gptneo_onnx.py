# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import forge
import torch
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify
from transformers import (
    AutoTokenizer,
    GPTNeoForCausalLM,
)
from test.utils import download_model
import onnx

# Wrapper to get around attention mask
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, None, attention_mask)


variants = [
    pytest.param(
        "EleutherAI/gpt-neo-125M",
        marks=pytest.mark.xfail,
    ),
    pytest.param(
        "EleutherAI/gpt-neo-1.3B",
        marks=pytest.mark.xfail,
    ),
    pytest.param(
        "EleutherAI/gpt-neo-2.7B",
        marks=pytest.mark.xfail,
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_gptneo_causal_lm_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.GPTNEO,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Load tokenizer and model
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant, return_dict=False, use_cache=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = download_model(GPTNeoForCausalLM.from_pretrained, variant)
    model.eval()
    torch_model = Wrapper(model)

    # Sample input text
    prompt = "My name is Bert, and I am"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, pad_to_max_length=True, truncation=True)
    inputs = [inputs["input_ids"], inputs["attention_mask"]]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/" + str(variant).split("/")[-1].replace("-", "_").replace(".", "_") + ".onnx"
    torch.onnx.export(torch_model, (inputs[0], inputs[1]), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    if variant in ["EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]:
        onnx.checker.check_model(onnx_path)
        framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)
        model = framework_model
    else:
        onnx.checker.check_model(onnx_model)
        framework_model = forge.OnnxModule(module_name, onnx_model)
        model = onnx_model

    # Compile model
    compiled_model = forge.compile(model, inputs, module_name=module_name)

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
