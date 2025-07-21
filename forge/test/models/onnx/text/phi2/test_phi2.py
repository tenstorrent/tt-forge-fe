# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import onnx
from transformers import AutoTokenizer, PhiForCausalLM

import forge
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties

from test.models.models_utils import TextModelWrapper

variants = ["microsoft/phi-2"]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.skip(reason="Transient test - Out of memory due to other tests in CI pipeline")
@pytest.mark.parametrize("variant", variants)
def test_phi2_clm_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.PHI2,
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.CAUSAL_LM,
    )

    # Load model and tokenizer from HuggingFace
    model = PhiForCausalLM.from_pretrained(variant, trust_remote_code=True, use_cache=False)
    framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
    framework_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(variant, trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # prepare input
    input_prompt = "Write a detailed analogy between mathematics and a lighthouse."
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True,
    )
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    inputs = [input_ids, attn_mask]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/model.onnx"
    torch.onnx.export(framework_model, tuple(inputs), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)

    # passing model file instead of model proto due to size of the model(>2GB) - #https://github.com/onnx/onnx/issues/3775#issuecomment-943416925
    onnx.checker.check_model(onnx_path)
    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Compile model
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
