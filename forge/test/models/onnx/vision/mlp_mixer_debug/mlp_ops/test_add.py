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



class Add0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, add_input_0, add_input_1):
        add_output_1 = forge.op.Add("", add_input_0, add_input_1)
        return add_output_1


class Add1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter("add1.weight_0", forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add1.weight_0"), add_input_1)
        return add_output_1


class Add2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add2_const_1", shape=(1,), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add2_const_1"))
        return add_output_1


class Add3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter("add3.weight_0", forge.Parameter(*(196,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add3.weight_0"), add_input_1)
        return add_output_1


class Add4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter("add4.weight_0", forge.Parameter(*(3072,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add4.weight_0"), add_input_1)
        return add_output_1


class Add5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter("add5.weight_0", forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add5.weight_0"), add_input_1)
        return add_output_1


class Add6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter("add6.weight_1", forge.Parameter(*(21843,), requires_grad=True, dev_data_format=forge.DataFormat.Float32))

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add6.weight_1"))
        return add_output_1



def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)

forge_modules_and_shapes_dtypes_list = [
    (Add0, [((1, 768, 14, 14), torch.float32), ((768, 1, 1), torch.float32)], {'model_names': ['onnx_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm'], 'pcc': 0.99}), 
    (Add1, [((1, 768, 384), torch.float32)], {'model_names': ['onnx_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm'], 'pcc': 0.99}), 
    (Add2, [((1, 768, 384), torch.float32)], {'model_names': ['onnx_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm'], 'pcc': 0.99}), 
    (Add3, [((1, 768, 196), torch.float32)], {'model_names': ['onnx_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm'], 'pcc': 0.99}), 
    (Add0, [((1, 196, 768), torch.float32), ((1, 196, 768), torch.float32)], {'model_names': ['onnx_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm'], 'pcc': 0.99}), 
    (Add4, [((1, 196, 3072), torch.float32)], {'model_names': ['onnx_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm'], 'pcc': 0.99}), 
    (Add2, [((1, 196, 3072), torch.float32)], {'model_names': ['onnx_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm'], 'pcc': 0.99}), 
    (Add5, [((1, 196, 768), torch.float32)], {'model_names': ['onnx_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm'], 'pcc': 0.99}), 
    (Add6, [((1, 21843), torch.float32)], {'model_names': ['onnx_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm'], 'pcc': 0.99}), 
]
@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
@pytest.mark.parametrize("training_test", [False, True], ids=["inference", "training"])
def test_module(forge_module_and_shapes_dtypes, training_test):
    
    record_forge_op_name("Add")
    
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
    
    
