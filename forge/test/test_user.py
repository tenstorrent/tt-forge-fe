# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Test "user experience" scenarios, i.e. different ways to use the API to run things on TT hardware
# Each test intentionally creates everything from scratch and uses no verification env, so that each
# of these tests can be used as user examples.
# There's also no verification of correctness of data, as that's not the point of these tests.
#
# All of these tests will run on silicon, in concurrent mode, by default. However, setting 
# FORGE_DEVMODE=1 env variable will drop them into Golden+sequential mode.

import queue
import torch
import forge
import pytest
from forge.config import _get_global_compiler_config

from forge.schedulers import LearningRateScheduler
from forge.forgeglobal import forge_reset
from test.utils import download_model

# https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
mp_context = torch.multiprocessing.get_context('spawn')

def _safe_read(q):
    """
    Read a queue, but return None if an error was raised in the meantime, preventing a hang on error.
    """
    while True:
        try:
            data = q.get(timeout = 0.5)
            return data
        except queue.Empty as _:
            if forge.error_raised():
                raise RuntimeError("Error raised in forge")
        except KeyboardInterrupt:
            return None

# Sample Forge module
class ForgeTestModule(forge.ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.weights1 = forge.Parameter(torch.rand(32, 32), requires_grad=True)
        self.weights2 = forge.Parameter(torch.rand(32, 32), requires_grad=True)

    def forward(self, act1, act2):
        m1 = forge.op.Matmul("matmul1", act1, self.weights1)
        m2 = forge.op.Matmul("matmul2", act2, self.weights2)
        return m1 + m2, m2

# Sample Forge module
class ForgeTestModuleOneOut(forge.ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.weights1 = forge.Parameter(torch.rand(32, 32), requires_grad=True)
        self.weights2 = forge.Parameter(torch.rand(32, 32), requires_grad=True)

    def forward(self, act1, act2):
        m1 = forge.op.Matmul("matmul1", act1, self.weights1)
        m2 = forge.op.Matmul("matmul2", act2, self.weights2)
        return m1 + m2

# Sample Forge module
class ForgeTestQueryKeyModule(forge.ForgeModule):
    def __init__(self, name, hidden_dim = 128, num_heads = 4):
        super().__init__(name)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.key_weights = forge.Parameter(torch.rand(1, 1, hidden_dim, hidden_dim), requires_grad=True)
        self.query_weights = forge.Parameter(torch.rand(1, 1, hidden_dim, hidden_dim), requires_grad=True)
        self.value_weights = forge.Parameter(torch.rand(1, 1, hidden_dim, hidden_dim), requires_grad=True)

    def forward(self, encoder_input):
        query = forge.op.Matmul(f"mha_query", encoder_input, self.query_weights)
        query = forge.op.HSlice(f"mha_query_slice", query, self.num_heads)

        key = forge.op.Matmul(f"mha_key", encoder_input, self.key_weights)
        key = forge.op.HSlice(f"mha_key_slice", key, self.num_heads)
        key = forge.op.Transpose(f"mha_key_transpose", key, 2, 3)

        attention_scores = forge.op.Matmul(f"mha_as", query, key)
        return attention_scores


class ForgeTestForkWithThreeUsers(forge.ForgeModule):
    def __init__(self, name, hidden_dim = 128, num_heads = 4):
        super().__init__(name)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.mm_a_weights = forge.Parameter(torch.rand(1, 1, hidden_dim, hidden_dim), requires_grad=True)
        self.mm_b_weights = forge.Parameter(torch.rand(1, 1, hidden_dim, hidden_dim), requires_grad=True)
        self.mm_c_weights = forge.Parameter(torch.rand(1, 1, hidden_dim, hidden_dim), requires_grad=True)

    def forward(self, encoder_input):
        a = forge.op.Matmul(f"mm_a", encoder_input, self.mm_a_weights)
        b = forge.op.Matmul(f"mm_b", encoder_input, self.mm_b_weights)
        c = forge.op.Matmul(f"mm_c", encoder_input, self.mm_c_weights)

        add_a_b = forge.op.Add(f"add_a_b", a, b)
        add_a_b_c = forge.op.Add(f"add_a_b_c", add_a_b, c)
        return add_a_b_c



# Sample PyTorch module
class PyTorchTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights1 = torch.nn.Parameter(torch.rand(32, 32), requires_grad=True)
        self.weights2 = torch.nn.Parameter(torch.rand(32, 32), requires_grad=True)

    def forward(self, act1, act2):
        m1 = torch.matmul(act1, self.weights1)
        m2 = torch.matmul(act2, self.weights2)
        return m1 + m2, m1

# Sample PyTorch module
class PyTorchTestModuleOneOut(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights1 = torch.nn.Parameter(torch.rand(32, 32), requires_grad=True)
        self.weights2 = torch.nn.Parameter(torch.rand(32, 32), requires_grad=True)

    def forward(self, act1, act2):
        m1 = torch.matmul(act1, self.weights1)
        m2 = torch.matmul(act2, self.weights2)
        return m1 + m2

class PyTorchTestModuleOneInputAndOneOut(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.rand(32, 32), requires_grad=True)
    
    def forward(self, act):
        m = torch.matmul(act, self.weights)
        return m

class PyTorchLoss(torch.nn.Module):
    def forward(self, input):
        return input.sum()

#
# Run inference on module directly
#
def test_module_direct_forge():
    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)

    # Run single inference pass on a Forge module directly
    output = ForgeTestModule("direct").run(input1, input2)
    print(output)

def test_module_direct_pytorch():
    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)

    # Run single inference pass on a PyTorch module, using a wrapper to convert to Forge first
    output = forge.PyTorchModule("direct_pt", PyTorchTestModule()).run(input1, input2)
    print(output)

#
# Run inference through run_inference without placing on device
#
def test_run_inference_direct_forge():
    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)

    # Run inference on a Forge module, with given inputs
    inputs = {"act2" : input2, "act1" : input1}
    output_q = forge.run_inference(ForgeTestModule("run_direct"), inputs=[inputs])
    output = _safe_read(output_q)
    print(output)

def test_run_inference_direct_pytorch():
    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)

    # Run inference, using a wrapper to convert PyTorch module to Forge, and with given inputs
    inputs = {"act2" : input2, "act1" : input1}
    output_q = forge.run_inference(forge.PyTorchModule("run_direct_pt", PyTorchTestModule()), inputs=[inputs])
    output = _safe_read(output_q)
    print(output)


#
# Run inference by placing on device first
#
def test_run_inference_placed_forge():
    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)

    # Create a TT device
    tt0 = forge.TTDevice("tt0")

    # Place a module on the device
    tt0.place_module(ForgeTestModule("placed"))

    # Push intputs to the device
    tt0.push_to_inputs((input1, input2))

    # Run pipeline, and read the outputs
    output_q = forge.run_inference()
    output = _safe_read(output_q)
    print(output)

def test_run_inference_placed_pytorch():
    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)

    # Create a TT device
    tt0 = forge.TTDevice("tt0")

    # Place a module on the device, using a wrapper to convert PyTorch module to Forge
    tt0.place_module(forge.PyTorchModule("placed_pt", PyTorchTestModule()))
    
    # Push intputs to the device
    tt0.push_to_inputs((input1, input2))

    # Run pipeline, and read the outputs
    output_q = forge.run_inference()
    output = _safe_read(output_q)
    print(output)

#
# Repeated calls to run inference on the same module
#
def test_module_direct_repeated():
    module = ForgeTestModule("direct")

    # Run on given inputs
    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)
    output = module.run(input1, input2)
    print(output)

    # Run again, without recompiling
    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)
    output = module.run(input1, input2)
    print(output)

    # Run again, without recompiling
    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)
    output = module.run(input1, input2)
    print(output)

def test_run_inference_placed_repeated():
    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)
    tt0 = forge.TTDevice("tt0")
    tt0.place_module(ForgeTestModule("placed"))

    # Push one input and run
    tt0.push_to_inputs((input1, input2))
    output_q = forge.run_inference()

    output = _safe_read(output_q)
    print(output)

    # Push two more inputs, and run one more time on both inputs, without recompiling
    for _ in range(2):
        input1 = torch.rand(4, 32, 32)
        input2 = torch.rand(4, 32, 32)
        tt0.push_to_inputs((input1, input2))

    forge.run_inference(input_count=2)

    for _ in range(2):
        output = _safe_read(output_q)
        print(output)


#
# Run inference through setup + run_forward calls
#
def test_setup_forward_calls():
    tt0 = forge.TTDevice("tt0")
    tt0.place_module(ForgeTestModule("placed"))

    # Compile & initialize the pipeline for inference, with given shapes
    output_q = forge.initialize_pipeline(training=False, sample_inputs=(torch.rand(4, 32, 32), torch.rand(4, 32, 32)))
        
    # Push & run_forward manually
    for _ in range(2):
        input1 = torch.rand(4, 32, 32)
        input2 = torch.rand(4, 32, 32)
        tt0.push_to_inputs((input1, input2))
        forge.run_forward(input_count=1)

        print(_safe_read(output_q))


#
# Run inference in concurrent mode, then push more inputs afterwards (won't work on Golden)
#
def test_run_inference_delayed_push():
    
    #### Skip the test on golden
    import os
    if "FORGE_DEVMODE" in os.environ:
        pytest.skip()
    ####

    tt0 = forge.TTDevice("tt0")
    tt0.place_module(ForgeTestModule("placed"))

    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)
    tt0.push_to_inputs((input1, input2))

    # Run with input count 3, although only one is pushed
    output_q = forge.run_inference(input_count=3)

    # Read one output that should've been produced
    output = _safe_read(output_q)
    print(output)

    # The inference thread is running in the background, waiting for data. Let's push two more.
    for _ in range(2):
        input1 = torch.rand(4, 32, 32)
        input2 = torch.rand(4, 32, 32)
        tt0.push_to_inputs((input1, input2))

    # Read two more outputs
    for _ in range(2):
        output = _safe_read(output_q)
        print(output)

#
# Run inference on multiple devices - combinations of cpu / tt device
#
def test_cpu_tt_pipeline():

    cpu0 = forge.CPUDevice("cpu0")
    cpu0.place_module(forge.PyTorchModule("stage0", PyTorchTestModule()))
    tt1 = forge.TTDevice("tt1")
    tt1.place_module(ForgeTestModule("stage1"))

    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)
    cpu0.push_to_inputs((input1, input2))

    output_q = forge.run_inference()
    print(_safe_read(output_q))

def test_cpu_tt_pipeline_compact():

    cpu0 = forge.CPUDevice("cpu0", module=forge.PyTorchModule("stage0", PyTorchTestModule()))
    tt1 = forge.TTDevice("tt1", module=ForgeTestModule("stage1"))

    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)
    cpu0.push_to_inputs((input1, input2))

    output_q = forge.run_inference()
    print(_safe_read(output_q))

# Run training, read back checkpoints and loss
def test_training_read_back():
    forge.config.set_configuration_options(
            default_df_override=forge.DataFormat.Float16_b,
    )
    tt0 = forge.TTDevice("tt0", module=ForgeTestModuleOneOut("module"))
    tt0.place_loss_module(forge.op.loss.L1Loss("l1_loss"))

    loss_q = mp_context.Queue()
    checkpoint_q = mp_context.Queue()

    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)
    tt0.push_to_inputs((input1, input2))
    tt0.push_to_target_inputs(torch.rand(4, 32, 32))

    forge.run_training(checkpoint_queue = checkpoint_q, loss_queue=loss_q)

    print("checkpoint: ", _safe_read(checkpoint_q))
    print("loss: ", _safe_read(loss_q))

# Run training pipeline, with loss on CPU, read back checkpoints and loss
#@pytest.mark.skip(reason="Intermittent hangs on silicon")
def test_training_pipeline_read_back():
    tt0 = forge.TTDevice("tt0", module=ForgeTestModule("stage0"))
    cpu1 = forge.CPUDevice("cpu1", module=forge.PyTorchModule("stage1", PyTorchTestModuleOneOut()))
    cpu1.place_loss_module(forge.PyTorchModule("l1loss", torch.nn.L1Loss()))

    loss_q = mp_context.Queue()
    checkpoint_q = mp_context.Queue()

    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)
    tt0.push_to_inputs((input1, input2))

    cpu1.push_to_target_inputs(torch.rand(4, 32, 32))

    forge.run_training(checkpoint_queue = checkpoint_q, loss_queue=loss_q)

    print("checkpoint: ", _safe_read(checkpoint_q))
    print("loss: ", _safe_read(loss_q))


#
# Run inference pipeline on a Transformers model
#
def test_transformers_pipeline_inference():

    from transformers import BertModel, BertTokenizer

    tokenizer = download_model(BertTokenizer.from_pretrained, "prajjwal1/bert-tiny")
    input_sentence = "BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was pretrained with two objectives: Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the sentence."
    input_tokens = tokenizer.encode(input_sentence, max_length=128, pad_to_max_length=True)

    model = download_model(BertModel.from_pretrained, "prajjwal1/bert-tiny", torchscript=False, add_pooling_layer=False)
    cpu0 = forge.CPUDevice("cpu0", module=forge.PyTorchModule("bert_embeddings", model.embeddings))
    tt0 = forge.TTDevice("tt1", module=forge.PyTorchModule("bert_encoder", model.encoder))

    cpu0.push_to_inputs(torch.Tensor(input_tokens).int().unsqueeze(0))
    output_q = forge.run_inference()

    print(_safe_read(output_q))

#
# Run inference pipeline on a Transformers model, enabling cpu fallback on unsupported ops
#
def test_transformers_pipeline_fallback_inference():

    from transformers import BertModel, BertTokenizer

    compiler_cfg = forge.config._get_global_compiler_config() 

    tokenizer = download_model(BertTokenizer.from_pretrained, "prajjwal1/bert-tiny")
    input_sentence = "BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was pretrained with two objectives: Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the sentence."
    input_tokens = tokenizer.encode(input_sentence, max_length=128, pad_to_max_length=True)

    model = download_model(BertModel.from_pretrained, "prajjwal1/bert-tiny", torchscript=False, add_pooling_layer=False)
    tt0 = forge.TTDevice("tt0", module=forge.PyTorchModule("bert", model))

    for i in range(5):
        tt0.push_to_inputs(torch.Tensor(input_tokens).int().unsqueeze(0))
        output_q = forge.run_inference()
        print(_safe_read(output_q))

#
# Run training through setup + manual loop of fwd/bwd/opt
#
def test_training_manual_loop_with_cpu_fallback():
    from transformers import BertForMaskedLM, BertTokenizer, BertConfig 

    config = download_model(BertConfig.from_pretrained, "prajjwal1/bert-tiny")
    model = BertForMaskedLM(config)
    tt0 = forge.TTDevice("tt0", module=forge.PyTorchModule("bert", model), optimizer=forge.optimizers.SGD(learning_rate=0.1, device_params=True))
    tt0.place_loss_module(forge.PyTorchModule("CEL", torch.nn.CrossEntropyLoss()))

    sample_inputs = (torch.randint(config.vocab_size, (1,128)) ,)
    sample_targets = (torch.rand(1, config.vocab_size) ,)

    checkpoint_q = forge.initialize_pipeline(
            training=True, 
            sample_inputs=sample_inputs,
            sample_targets=sample_targets)


    for step in range(2):
        for acc_step in range(2):
            tt0.push_to_inputs(torch.randint(config.vocab_size, (1,128)))
            tt0.push_to_target_inputs(torch.rand(1, config.vocab_size).long())
            forge.run_forward(input_count = 1)
            forge.run_backward(input_count = 1, zero_grad = (acc_step == 0))

        forge.run_optimizer(checkpoint=True)

# Run training through run_training without placing on device
# Run training by placing on device first
# Repeated calls to run training
# Run training in concurrent mode, then push inputs afterwards
# Run training in concurrent mode, read checkpoints as they come out
# Run inference on multiple devices - combinations of cpu / tt device

#
# Run training through setup + manual loop of fwd/bwd/opt
#
def test_training_manual_loop():

    tt0 = forge.TTDevice("tt0", module=ForgeTestModule("stage0"), optimizer=forge.optimizers.SGD(learning_rate=0.1, device_params=True))
    cpu1 = forge.CPUDevice("cpu1", module=forge.PyTorchModule("stage1", PyTorchTestModuleOneOut()),
            optimizer_f = lambda m: torch.optim.SGD(m.parameters(), lr=0.5))
    cpu1.place_loss_module(forge.PyTorchModule("l1loss", torch.nn.L1Loss()))
    
    # Compile & initialize the pipeline for training, with given shapes
    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)
    checkpoint_q = forge.initialize_pipeline(
            training=True, 
            sample_inputs=(input1, input2),
            sample_targets=(torch.rand(4, 32, 32),))


    for step in range(2):
        for acc_step in range(2):
            tt0.push_to_inputs((input1, input2))
            cpu1.push_to_target_inputs(torch.rand(4, 32, 32))

            forge.run_forward(input_count = 1)
            forge.run_backward(input_count = 1, zero_grad = (acc_step == 0))

        forge.run_optimizer(checkpoint=True)

    print("Checkpoint: ", _safe_read(checkpoint_q))

#
# Run training through setup + manual loop of fwd/bwd, while copying final gradients
#
def test_training_manual_loop_no_opt():

    #### Skip the test on golden. It should work, need to debug why it doesn't.
    import os
    if "FORGE_DEVMODE" in os.environ:
        pytest.skip()
    ####

    tt0 = forge.TTDevice("tt0", module=ForgeTestModule("stage0"))
    cpu1 = forge.CPUDevice("cpu1", module=forge.PyTorchModule("stage1", PyTorchTestModuleOneOut()))
    cpu1.place_loss_module(forge.PyTorchModule("l1loss", torch.nn.L1Loss()))
    
    # Compile & initialize the pipeline for training, with given shapes
    forge.initialize_pipeline(
            training=True, 
            sample_inputs=(torch.rand(4, 32, 32), torch.rand(4, 32, 32)), 
            sample_targets=(torch.rand(4, 32, 32),))

    steps = 2

    for step in range(steps):
        for acc_step in range(1):
    
            input1 = torch.rand(4, 32, 32)
            input2 = torch.rand(4, 32, 32)
            tt0.push_to_inputs((input1, input2))

            cpu1.push_to_target_inputs(torch.rand(4, 32, 32))

            forge.run_forward(input_count = 1)
            forge.run_backward(input_count = 1, zero_grad = (acc_step == 0))

        print("Gradients on step ", step, ": ", forge.get_parameter_gradients())

#
# Run training and upload new weights from host
#
def test_training_weight_update_on_host():

    #### Skip the test on golden. It should work, need to debug why it doesn't.
    import os
    if "FORGE_DEVMODE" in os.environ:
        pytest.skip()
    ####

    tt0 = forge.TTDevice("tt0", module=ForgeTestModule("stage0"))
    cpu1 = forge.CPUDevice("cpu1", module=forge.PyTorchModule("stage1", PyTorchTestModuleOneOut()))
    cpu1.place_loss_module(forge.PyTorchModule("l1loss", torch.nn.L1Loss()))
    
    # Compile & initialize the pipeline for training, with given shapes
    forge.initialize_pipeline(training=True, 
            sample_inputs=(torch.rand(4, 32, 32), torch.rand(4, 32, 32)), 
            sample_targets=(torch.rand(4, 32, 32),))

    for _ in range(2):
        input1 = torch.rand(4, 32, 32)
        input2 = torch.rand(4, 32, 32)
        tt0.push_to_inputs((input1, input2))

        cpu1.push_to_target_inputs(torch.rand(4, 32, 32))

    # Run fwd/bwd to calculate parameter gradients
    forge.run_forward(input_count = 1)
    forge.run_backward(input_count = 1, zero_grad = True)

    # Retrieve weights and gradients, and use host optimizer to update weights
    grads = forge.get_parameter_gradients(tt0)
    params = forge.get_parameter_checkpoint(tt0)
    for name in params[0]:
        params[0][name].value().grad = grads[0][name].value()
    opt = torch.optim.SGD([p.value() for p in params[0].values()], lr=10.0)
    opt.step()

    # Push new weights to the device
    forge.update_device_parameters(tt0, params)

    # Run again with new weights
    forge.run_forward(input_count = 1)
    forge.run_backward(input_count = 1, zero_grad = True)

# 
# Run inference pipeline and provide mp queues for device-to-device data
#
def test_inference_device_to_device_data():
    tt0 = forge.TTDevice("tt0", module=ForgeTestModule("stage0"))
    cpu1 = forge.CPUDevice("cpu1", module=forge.PyTorchModule("stage1", PyTorchTestModule()))
    cpu2 = forge.CPUDevice("cpu2", module=forge.PyTorchModule("stage2", PyTorchTestModuleOneOut()))
    
    # Compile & initialize the pipeline for inference, and provide d2d mp queues to store device-to-device data in for further analysis
    tt0_output_q = mp_context.Queue()
    cpu1_output_q = mp_context.Queue()
    forge.initialize_pipeline(training=False, d2d_fwd_queues=[tt0_output_q, cpu1_output_q], 
            sample_inputs=(torch.rand(4, 32, 32), torch.rand(4, 32, 32) ))

    for _ in range(2):
        input1 = torch.rand(4, 32, 32)
        input2 = torch.rand(4, 32, 32)
        tt0.push_to_inputs((input1, input2))

    # Run fwd
    forge.run_forward(input_count = 1)

    # Read d2d queues
    print(_safe_read(tt0_output_q))
    print(_safe_read(cpu1_output_q))

# 
# Run training pipeline and provide mp queues for device-to-device data
#

def test_training_device_to_device_data():
    
    tt0 = forge.TTDevice("tt0", module=ForgeTestModule("stage0"))
    cpu1 = forge.CPUDevice("cpu1", module=forge.PyTorchModule("stage1", PyTorchTestModule()))
    cpu2 = forge.CPUDevice("cpu2", module=forge.PyTorchModule("stage2", PyTorchTestModuleOneOut()))
    cpu2.place_loss_module(forge.PyTorchModule("l1loss", torch.nn.L1Loss()))
    
    # Compile & initialize the pipeline for inference, and provide d2d mp queues to store device-to-device data in for further analysis
    tt0_output_q = mp_context.Queue()
    cpu1_output_q = mp_context.Queue()
    cpu1_bwd_output_q = mp_context.Queue()
    cpu2_bwd_output_q = mp_context.Queue()
    forge.initialize_pipeline(
            training=True, 
            d2d_fwd_queues=[tt0_output_q, cpu1_output_q], 
            d2d_bwd_queues=[cpu1_bwd_output_q, cpu2_bwd_output_q], 
            sample_inputs=(torch.rand(4, 32, 32), torch.rand(4, 32, 32)), 
            sample_targets=(torch.rand(4, 32, 32),))

    for _ in range(2):
        input1 = torch.rand(4, 32, 32)
        input2 = torch.rand(4, 32, 32)
        tt0.push_to_inputs((input1, input2))

        cpu2.push_to_target_inputs(torch.rand(4, 32, 32))

    # Run fwd/bwd 
    forge.run_forward()
    forge.run_backward(zero_grad = True)

    # Read d2d queues
    print(_safe_read(tt0_output_q))
    print(_safe_read(cpu1_output_q))
    print(_safe_read(cpu1_bwd_output_q))
    print(_safe_read(cpu2_bwd_output_q))
    forge.get_parameter_gradients(tt0)

#
# Override data formats
#
def test_data_formats_input_override():

    mod = ForgeTestModule("mod")
    tt0 = forge.TTDevice("tt0", module=mod)

    # Explicitly set data formats for parameters and inputs
    mod.weights1.set_data_format(forge.DataFormat.Float16)
    mod.weights2.set_data_format(forge.DataFormat.Float16)
    input1 = torch.rand(4, 32, 32, dtype=torch.float16)
    input2 = torch.rand(4, 32, 32, dtype=torch.float16)
    tt0.push_to_inputs((input1, input2))

    forge.run_inference()

def test_data_formats_fp32_fallback():
    
    # On this device, fall back to Float16 wherever Float32 is used
    tt0 = forge.TTDevice("tt0", module=ForgeTestModule("mod"), fp32_fallback=forge.DataFormat.Float16)

    # Push Float32, which will be converted to Float16 due to fp32_fallback
    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)
    tt0.push_to_inputs((input1, input2))

    forge.run_inference()

def test_data_formats_op_override():
    
    tt0 = forge.TTDevice("tt0", module=ForgeTestModule("mod"))

    # Use API to set manual data format override on an op
    forge.configure_mixed_precision(name_regex="matmul1", output_df=forge.DataFormat.Bfp8_b)
    
    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)
    tt0.push_to_inputs((input1, input2))

    forge.run_inference()

class TorchSchedulerWithWarmupAndDecay(forge.torch_schedulers.TorchLearningRateScheduler):
    def __init__(self, optimizer):
        super().__init__(optimizer)
    
    def get_lr(self):
        return [self.optimizer.param_groups[0]["lr"] + 1]
    
    def step(self):
        super().step()
        print(f"Torch optimizer learning rate updated to {self.optimizer.param_groups[0]['lr']}")


class TestScheduler(LearningRateScheduler):
        def __init__(self, optimizer):
            super().__init__(optimizer)
        
        def get_lr(self):
            return self.optimizer.learning_rate + 1
        
        def step(self):
            super().step()
            print(f"Forge optimizer learning rate updated to {self.optimizer.learning_rate}")
        
        def get_pytorch_scheduler(self, optimizer: torch.optim.Optimizer):
            if self.torch_scheduler is None:
                self.torch_scheduler = TorchSchedulerWithWarmupAndDecay(
                    optimizer=optimizer
                )
            
            return self.torch_scheduler


# Run the learning rate scheduler across 100 steps to
# show how optimizer learning rate gets updated
def test_learning_rate_scheduler():
            
    lr = 1
    optimizer = forge.optimizers.SGD(learning_rate=lr, device_params=True)
    scheduler = TestScheduler(optimizer=optimizer)
    
    tt0 = forge.TTDevice(
        "tt0", 
        module=ForgeTestModuleOneOut("stage0"), 
        optimizer=optimizer,
        scheduler=scheduler
    )
    cpu1 = forge.CPUDevice(
        "cpu1",
        module=forge.PyTorchModule(
            "stage1",
            PyTorchTestModuleOneInputAndOneOut()
        ),
        optimizer_f=lambda module: torch.optim.SGD(module.parameters(), lr=lr),
        scheduler_f=lambda optimizer: scheduler.get_pytorch_scheduler(optimizer)
    )
    cpu1.place_loss_module(
        forge.PyTorchModule(
            "loss",
            PyTorchLoss()
        )
    )

    sequential = True
    forge.initialize_pipeline(training=True, 
            sample_inputs=(torch.rand(4, 32, 32), torch.rand(4, 32, 32)), 
            sample_targets=(torch.rand(4, 32, 32),), _sequential=sequential)

    for _ in range(100):
        forge.run_schedulers(sequential)
    
    
    
def test_specific_chip_id():
    """
    Run inference on a specific chip on a multi-chip system
    """
    num_devices = len(forge.detect_available_devices())

    if num_devices < 2:
        pytest.skip("Need at least 2 devices to run chip-id test")

    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)

    # Create a TT device, on last available chip
    tt0 = forge.TTDevice("tt0", chip_ids=[num_devices-1])

    # Place a module on the device
    tt0.place_module(ForgeTestModule("last_chip"))

    # Push intputs to the device
    tt0.push_to_inputs((input1, input2))

    # Run pipeline, and read the outputs
    output_q = forge.run_inference()
    output = _safe_read(output_q)
    print(output)

def _run_on_chip(chip_id: int):

    # Each process needs to have its own temporary dir
    forge.set_configuration_options(backend_output_dir=f"tt_build/test_out_chip_{chip_id}")

    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)

    # Create a TT device, on last available chip
    tt0 = forge.TTDevice("tt0", chip_ids=[chip_id])

    # Place a module on the device
    tt0.place_module(ForgeTestModule(f"chip_{chip_id}"))

    # Push intputs to the device
    tt0.push_to_inputs((input1, input2))

    # Run pipeline, and read the outputs
    output_q = forge.run_inference()
    output = _safe_read(output_q)
    print("From chip ", chip_id, ":", output)

    # Clean up the process so we can end it cleanly
    forge.shutdown()


def test_parallel_chips():
    """
    Run different models on multiple chips at the same time
    """
    pytest.skip("Appears to hang now")
    num_devices = len(forge.detect_available_devices())

    if num_devices < 2:
        pytest.skip("Need at least 2 devices to run parallel chip test")

    procs = []
    for i in range(num_devices):
        p = mp_context.Process(target=_run_on_chip, args=(i,))
        p.start()
        procs.append(p)

    for i, p in enumerate(procs):
        p.join()

# def test_tti_inference_save_and_load():
#     available_devices = forge.detect_available_devices()
#     if available_devices and available_devices[0] == BackendDevice.Grayskull:
#         tt0 = forge.TTDevice(
#             "tt0",
#             arch=BackendDevice.Grayskull,
#             devtype=BackendType.Golden,
#         )
#     else:
#         tt0 = forge.TTDevice(
#             "tt0",
#             arch=BackendDevice.Wormhole_B0,
#             devtype=BackendType.Golden,
#         )


#     module = ForgeTestModule("test_forge_module")
#     tt0.place_module(module)

#     # Saving to Archive
#     input_shape = (1, 1, 32, 32)
#     input1, input2  = torch.rand(*input_shape), torch.rand(*input_shape)
#     device_img = tt0.compile_to_image(
#         img_path="device_images/test_tt0.tti", 
#         training=False,
#         sample_inputs=(input1, input2),
#     )
#     forge_reset()  # flush the global state that lingers around for test

#     # Loading from Archive
#     tt1 = forge.TTDevice.load_image(img_path="device_images/test_tt0.tti")
#     tt1.push_to_inputs((input1, input2))
#     output_q = forge.run_inference()
#     output = _safe_read(output_q)


@pytest.mark.parametrize("hoist_tms", [True, False])
def test_nop_insertion_api(hoist_tms):
    tt0 = forge.TTDevice("tt0", module=ForgeTestQueryKeyModule(f"query_key_module_hoist_tms_{hoist_tms}"))

    # Use API to set manual data format override on an op
    forge.insert_nop("mha_key", "mha_as", hoist_tms=hoist_tms)
    microbatch_size, seq_len, hidden_dim = (1, 128, 128)
    encoder_input = torch.rand(microbatch_size, seq_len, hidden_dim)

    tt0.push_to_inputs((encoder_input))
    forge.run_inference()

@pytest.mark.parametrize("hoist_tms", [True, False])
def test_nop_fork_insertion_api(hoist_tms):
    tt0 = forge.TTDevice("tt0", module=ForgeTestQueryKeyModule(f"forking_nop_insertion{hoist_tms}"))

    # Use API to set manual data format override on an op
    forge.insert_nop("encoder_input", ["mha_key", "mha_query"], hoist_tms=hoist_tms)
    microbatch_size, seq_len, hidden_dim = (1, 128, 128)
    encoder_input = torch.rand(microbatch_size, seq_len, hidden_dim)

    tt0.push_to_inputs((encoder_input))
    forge.run_inference()

@pytest.mark.parametrize("hoist_tms", [True, False])
def test_nop_daily_chain_insertion_api(hoist_tms):
    tt0 = forge.TTDevice("tt0", module=ForgeTestForkWithThreeUsers(f"daisy_chain_nop_insertion{hoist_tms}"))

    # Use API to set manual data format override on an op
    forge.insert_nop("encoder_input", ["mm_a", "mm_b", "mm_c"], hoist_tms=hoist_tms)
    forge.insert_nop("buffer_0_encoder_input_mm_a", ["mm_b", "mm_c"], hoist_tms=hoist_tms)
    forge.insert_nop("buffer_0_buffer_0_encoder_input_mm_a_mm_b", ["mm_c"], hoist_tms=hoist_tms)
    microbatch_size, seq_len, hidden_dim = (1, 128, 128)
    encoder_input = torch.rand(microbatch_size, seq_len, hidden_dim)

    tt0.push_to_inputs((encoder_input))
    forge.run_inference()

def test_dram_channel_override():
    tt0 = forge.TTDevice("tt0", module=ForgeTestModule(f"dram_channel_override"))

    # Use API to set manual data format override on an op
    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)
    forge.config.override_dram_queue_placement("e2e_matmul1_0", channel=0)
    forge.config.set_epoch_break("matmul2")

    tt0.push_to_inputs((input1, input2))
    forge.run_inference()

@pytest.mark.parametrize("loss", ["l1", "mse"])
def test_loss_module_on_ttdevice(loss):
    import torch.nn as nn
    class Lin(nn.Module):
        def __init__(self, d_model):
            super(Lin, self).__init__()
            self.input_linear = nn.Linear(1, d_model)

        def forward(self, src):
            output = self.input_linear(src)
            return output

    model = Lin(1)
    tt0 = forge.TTDevice(
        "tt0",
        module=forge.PyTorchModule("lin", model),
        optimizer=forge.optimizers.SGD(learning_rate=0.1, device_params=True)
    )
    if loss == "mse":
        tt0.place_loss_module(forge.PyTorchModule("mse_loss", nn.MSELoss()))
    else:
        tt0.place_loss_module(forge.PyTorchModule("l1_loss", nn.L1Loss()))

    inputs = torch.rand(1, 1)
    targets = torch.rand(1, 1)

    # Initialize pipeline
    checkpoint_q = forge.initialize_pipeline(
       training=True,
       sample_inputs=(inputs,),
       sample_targets=(targets,)
    )

    tt0.push_to_inputs(inputs)
    tt0.push_to_target_inputs(targets)
    forge.run_forward(input_count=1)
    forge.run_backward(input_count=1, zero_grad=True)
    forge.run_optimizer(checkpoint=True)

