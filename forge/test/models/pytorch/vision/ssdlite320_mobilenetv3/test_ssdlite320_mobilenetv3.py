# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from third_party.tt_forge_models.ssdlite320_mobilenetv3.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.vision.ssdlite320_mobilenetv3.model_utils.model_utils import (
    Wrapper,
)

variants_with_weights = {
    "ssdlite320_mobilenet_v3_large": ModelVariant.SSDLITE320_MOBILENET_V3_LARGE,
}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants_with_weights.keys())
def test_ssdlite320_mobilenetv3(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.SSDLITE320MOBILENETV3,
        variant=variant,
        task=Task.OBJECT_DETECTION,
        source=Source.TORCHVISION,
    )
    pytest.xfail(reason="Fatal Python error: Aborted")

    # Load model and input using the new loader
    variant_enum = variants_with_weights[variant]
    loader = ModelLoader(variant=variant_enum)

    # Load model with bfloat16 override
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    framework_model = Wrapper(framework_model)

    # Load inputs with bfloat16 override
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[inputs],
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify([inputs], framework_model, compiled_model)
