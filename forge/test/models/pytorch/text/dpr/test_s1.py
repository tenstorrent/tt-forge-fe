# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch

from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.utils import download_model

from transformers import (
    DPRReader,
)
import forge

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.span_predictor.qa_classifier

    def forward(self, x):
        y = self.model(x)
        return y


variants = ["facebook/dpr-reader-single-nq-base"]

@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dpr_reader_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DPR,
        variant=variant,
        suffix="reader",
        source=Source.HUGGINGFACE,
        task=Task.QA,
    )

    model_ckpt = variant

    framework_model = download_model(DPRReader.from_pretrained, model_ckpt, return_dict=False)
    framework_model = Wrapper(framework_model)

    inputs = [torch.randn(1,768)]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

   