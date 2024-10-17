import forge
# from forge.verify.backend import verify_module
# from forge import VerifyConfig
# from forge.verify import TestKind

import sys

sys.path.append("tt-forge-fe/forge/test/model_demos/high_prio/cnn/pytorch/model2/pytorch")

from nbeats.scripts import get_electricity_dataset_input, NBeatsWithGenericBasis, NBeatsWithTrendBasis, NBeatsWithSeasonalityBasis


def test_nbeats_with_seasonality_basis(test_device):
    # PyBuda configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = forge._C.Float16_b
    compiler_cfg.enable_auto_fusing = False

    x, x_mask = get_electricity_dataset_input()

    pytorch_model = NBeatsWithSeasonalityBasis(
        input_size=72,
        output_size=24,
        num_of_harmonics=1,
        stacks=30,
        layers=4,
        layer_size=2048,
    )
    pytorch_model.eval()

    # Create forge.PyTorchModule using the loaded Pytorch model
    # tt_model = forge.PyTorchModule("nbeats_seasonality", pytorch_model)
    

    pcc = 0.99
    compiled_model = forge.compile(pytorch_model, sample_inputs=[x, x_mask])
    
    # verify_module(
    #     tt_model,
    #     input_shapes=[(x.shape, x_mask.shape)],
    #     inputs=[(x, x_mask)],
    #     verify_cfg=VerifyConfig(
    #         arch=test_device.arch,
    #         devtype=test_device.devtype,
    #         devmode=test_device.devmode,
    #         test_kind=TestKind.INFERENCE,
    #         pcc=pcc,
    #     ),
    # )



def test_nbeats_with_generic_basis(test_device):
    # PyBuda configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = forge._C.Float16_b

    x, x_mask = get_electricity_dataset_input()

    pytorch_model = NBeatsWithGenericBasis(
        input_size=72, output_size=24, stacks=30, layers=4, layer_size=512
    )
    pytorch_model.eval()

    # Create forge.PyTorchModule using the loaded Pytorch model
    # tt_model = forge.PyTorchModule("nbeats_generic", pytorch_model)

    pcc = 0.99
    compiled_model = forge.compile(pytorch_model, sample_inputs=[x, x_mask])
    # verify_module(
    #     tt_model,
    #     input_shapes=[(x.shape, x_mask.shape)],
    #     inputs=[(x, x_mask)],
    #     verify_cfg=VerifyConfig(
    #         arch=test_device.arch,
    #         devtype=test_device.devtype,
    #         devmode=test_device.devmode,
    #         test_kind=TestKind.INFERENCE,
    #         pcc=pcc,
    #     ),
    # )


def test_nbeats_with_trend_basis(test_device):
    # PyBuda configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = forge._C.Float16_b

    # if test_device.arch == forge.BackendDevice.Grayskull:
    #     compiler_cfg.enable_auto_fusing = False

    x, x_mask = get_electricity_dataset_input()

    pytorch_model = NBeatsWithTrendBasis(
        input_size=72,
        output_size=24,
        degree_of_polynomial=3,
        stacks=30,
        layers=4,
        layer_size=256,
    )
    pytorch_model.eval()

    # Create forge.PyTorchModule using the loaded Pytorch model
    compiled_model = forge.compile(pytorch_model, sample_inputs=[x, x_mask])
    # tt_model = forge.PyTorchModule("nbeats_trend", pytorch_model)

    # pcc = 0.99

    # verify_module(
    #     tt_model,
    #     input_shapes=[(x.shape, x_mask.shape)],
    #     inputs=[(x, x_mask)],
    #     verify_cfg=VerifyConfig(
    #         arch=test_device.arch,
    #         devtype=test_device.devtype,
    #         devmode=test_device.devmode,
    #         test_kind=TestKind.INFERENCE,
    #         pcc=pcc,
    #     ),
    # )
