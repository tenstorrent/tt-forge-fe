import forge
import forge.op
from forge import ForgeModule

from loguru import logger
import torch

from forge import Tensor, compile
from forge.verify.verify import verify
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.config import VerifyConfig
from forge.forge_property_utils import record_forge_op_name, record_op_model_names, record_forge_op_args, record_single_op_operands_info
import pytest



class Conv2D0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter("conv2d0.weight_1", forge.Parameter(*(768, 3, 16, 16), requires_grad=True, dev_data_format=forge.DataFormat.Float32))

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d("", conv2d_input_0, self.get_parameter("conv2d0.weight_1"), stride=[16, 16], padding=[0, 0, 0, 0], dilation=1, groups=1, channel_last=False)
        return conv2d_output_1



def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)

forge_modules_and_shapes_dtypes_list = [
    (Conv2D0, [((1, 3, 224, 224), torch.float32)], {'model_names': ['onnx_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm'], 'pcc': 0.99, 'args': {'stride': '[16, 16]', 'padding': '[0, 0, 0, 0]', 'dilation': '1', 'groups': '1', 'channel_last': 'False'}}), 
]
@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
@pytest.mark.parametrize("training_test", [False, True], ids=["inference", "training"])
def test_module(forge_module_and_shapes_dtypes, training_test):
    
    record_forge_op_name("Conv2d")
    
    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes
    
    pcc = metadata.get("pcc")
    
    for metadata_name, metadata_value in metadata.items():
        if metadata_name in ['pcc']:
            continue
        elif metadata_name == "model_names":
            record_op_model_names(metadata_value)
        elif metadata_name == "args":
            record_forge_op_args(metadata_value)
        else:
            logger.warning("No utility function available in forge property handler to record %s property", metadata_name)
    
    max_int = 1000
    inputs = [Tensor.create_from_shape(operand_shape, operand_dtype, max_int=max_int, requires_grad = training_test) for operand_shape, operand_dtype in operand_shapes_dtypes]
    
    framework_model = forge_module(forge_module.__name__)
    
    for name, parameter in framework_model._parameters.items():
        parameter_tensor = Tensor.create_torch_tensor(shape=parameter.shape.get_pytorch_shape(), dtype=parameter.pt_data_format, max_int=max_int, requires_grad = training_test)
        framework_model.set_parameter(name, parameter_tensor)
    
    for name, constant in framework_model._constants.items():
        constant_tensor = Tensor.create_torch_tensor(shape=constant.shape.get_pytorch_shape(), dtype=constant.pt_data_format, max_int=max_int, requires_grad = training_test)
        framework_model.set_constant(name, constant_tensor)
    
    record_single_op_operands_info(framework_model, inputs)
    
    compiler_cfg = forge.config.CompilerConfig()
    if "default_df_override" in metadata.keys():
        compiler_cfg.default_df_override = forge.DataFormat.from_json(metadata["default_df_override"])
    
    compiled_model = compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg, training=training_test)
    
    verify(inputs, framework_model, compiled_model, with_backward=training_test, verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
    
    
