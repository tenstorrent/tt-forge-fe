# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import onnx
import forge

from test.models.pytorch.multimodal.stable_diffusion.utils.model import (
    load_pipe_v35,
    stable_diffusion_preprocessing_v35,
)
from test.models.utils import Framework, Source, Task, build_module_name


class StableDiffusionWrapper(torch.nn.Module):
    def __init__(self, model, joint_attention_kwargs=None, return_dict=False):
        super().__init__()
        self.model = model
        self.joint_attention_kwargs = joint_attention_kwargs
        self.return_dict = return_dict

    def forward(self, latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds):
        noise_pred = self.model(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=self.return_dict,
        )[0]
        return noise_pred


@pytest.mark.nightly
# @pytest.mark.skip(
#     reason="Insufficient host DRAM to run this model (requires a bit more than 40 GB during compile time)"
# )
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "stable-diffusion-3.5-medium",
            #         marks=pytest.mark.xfail(reason="Exception: warning unhandled case: <class 'NoneType'>"),
            #     ),
            #     pytest.param(
            #         "stable-diffusion-3.5-large",
            #         marks=pytest.mark.xfail(reason="Exception: warning unhandled case: <class 'NoneType'>"),
            #     ),
            #     pytest.param(
            #         "stable-diffusion-3.5-large-turbo",
            #         marks=pytest.mark.xfail(reason="Exception: warning unhandled case: <class 'NoneType'>"),
        ),
    ],
)
def test_stable_diffusion_v35(forge_property_recorder, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="stable_diffusion",
        variant=variant,
        task=Task.CONDITIONAL_GENERATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("priority")
    forge_property_recorder.record_model_name(module_name)

    # Load pipeline
    pipe = load_pipe_v35(variant)

    # Extract only the transformer, as the forward pass occurs here.
    framework_model = pipe.transformer

    # TODO: Implement post-processing using VAE decode after obtaining the transformer output.
    framework_model = StableDiffusionWrapper(framework_model, joint_attention_kwargs=None, return_dict=False)

    # Load inputs
    prompt = "An astronaut riding a green horse"
    latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds = stable_diffusion_preprocessing_v35(pipe, prompt)
    inputs = [latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds]
    # Export model to ONNX
    onnx_path = f"{tmp_path}/stable_diffusion_v35.onnx"
    torch.onnx.export(
        framework_model,
        (latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds),
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
    )

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)
