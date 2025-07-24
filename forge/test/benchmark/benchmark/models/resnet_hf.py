# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import pytest
import time
from datetime import datetime

# Third-party modules
import torch
from transformers import ResNetForImageClassification

# Forge modules
import forge
from forge.verify.value_checkers import AutomaticValueChecker
from forge._C.runtime.experimental import configure_devices, DeviceSettings
from forge._C import DataFormat
from forge.config import CompilerConfig, MLIRConfig
from test.utils import download_model


def test_resnet_hf():
    batch_size = 8
    input_size = (224, 224)
    channel_size = 3
    loop_count = 32
    variant = "microsoft/resnet-50"

    torch.manual_seed(1)
    # Random data
    inputs = [torch.rand(batch_size, channel_size, *input_size)]
    inputs = [item.to(torch.bfloat16) for item in inputs]

    framework_model = download_model(
        ResNetForImageClassification.from_pretrained, variant, return_dict=False, torch_dtype=torch.bfloat16
    ).to(dtype=torch.bfloat16)

    # Configure compiler
    compiler_cfg = CompilerConfig()
    compiler_cfg.default_df_override = DataFormat.Float16_b
    # Turn on MLIR optimizations
    compiler_cfg.mlir_config = (
        MLIRConfig()
        .set_enable_optimizer(True)
        .set_enable_fusing(True)
        .set_enable_fusing_conv2d_with_multiply_pattern(True)
        .set_enable_memory_layout_analysis(False)
    )

    # Compile model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs[0], compiler_cfg=compiler_cfg)

    # Enable program cache on all devices
    settings = DeviceSettings()
    settings.enable_program_cache = True
    configure_devices(device_settings=settings)

    # Run the model 3 times for warmup.
    for _ in range(3):
        compiled_model(inputs[0])

    start = time.time()
    for i in range(loop_count):
        co_out = compiled_model(inputs[0])[0]
    end = time.time()
    evaluation_score = 0.0

    date = datetime.now().strftime("%d-%m-%Y")
    total_time = end - start
    total_samples = batch_size * loop_count

    samples_per_sec = total_samples / total_time
    model_name = "Resnet 50 HF"
    model_type = "Classification"

    model_type += ", Random Input Data"
    dataset_name = model_name + ", Random Data"
    num_layers = 50  # Number of layers in the model, in this case 50 layers

    print("====================================================================")
    print("| Resnet Benchmark Results:                                        |")
    print("--------------------------------------------------------------------")
    print(f"| Model: {model_name}")
    print(f"| Model type: {model_type}")
    print(f"| Dataset name: {dataset_name}")
    print(f"| Date: {date}")
    print(f"| Total execution time: {total_time}")
    print(f"| Total samples: {total_samples}")
    print(f"| Sample per second: {samples_per_sec}")
    print(f"| Evaluation score: {evaluation_score}")
    print(f"| Batch size: {batch_size}")
    print(f"| Data format: {torch.bfloat16}")
    print(f"| Input size: {input_size}")
    print(f"| Channel size: {channel_size}")
    print("====================================================================")

    fw_out = framework_model(inputs[-1])[0]
    co_out = co_out.to("cpu")
    AutomaticValueChecker(pcc=0.94).check(fw_out=fw_out, co_out=co_out)

    return {}
