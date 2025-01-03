
import pytest
from test.mlir.mnist.utils import MNISTLinear
import torch
import torch.nn as nn
from forge.verify.compare import compare_with_golden
import forge


def copy_model_params(model1: nn.Module, model2: nn.Module):
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        param2.data = param1.data.clone()

def get_gradients(model):
    return [p.grad for p in model.parameters()][::-1]


def train_and_compare_models(num_epochs, batch_size, shape, loss_fn, tt_model, tt_optimizer, golden_model, golden_optimizer):
    for epoch in range(num_epochs):
        # Forward pass
        x = torch.randn(batch_size, shape[0])
        y = torch.randn(batch_size, shape[1])
        
        tt_out = tt_model(x)
        gold_out = golden_model(x)
        
        loss = loss_fn(gold_out, y)
        loss.backward()
        grads = get_gradients(golden_model)
        
        # Copy the gradients to the tt model because only the optimizer should be tested
        tt_model.gradient_outputs = grads
        tt_optimizer.step()
        golden_optimizer.step()

        # Compare all the parameters
        for i, (tt_param, golden_param) in enumerate(zip(tt_model.framework_module.module.parameters(), golden_model.parameters())):
            assert compare_with_golden(tt_param, golden_param, pcc=0.95), f"Weight mismatch at epoch {epoch}\n {tt_param}, {golden_param} and param {i}"

        print(f"Epoch: {epoch}, Loss: {loss.item()}")


@pytest.mark.parametrize(
    "shape",
    [
        (784, 10),
        (33, 127),
        (128, 20),
    ],
)
def test_sgd(shape):
    torch.manual_seed(0)
    num_epochs = 10
    learning_rate = 0.01
    batch_size = 64

    framework_model = MNISTLinear(input_size=shape[0], output_size=shape[1], bias=False)
    golden_model = MNISTLinear(input_size=shape[0], output_size=shape[1], bias=False)
    
    copy_model_params(framework_model, golden_model)

    tt_optimizer = forge.optimizers.SGD(learning_rate=learning_rate)
    golden_optimizer = torch.optim.SGD(golden_model.parameters(), lr=learning_rate)

    tt_model = forge.compile(
        framework_model, 
        sample_inputs=[torch.randn(batch_size, shape[0])],
        optimizer=tt_optimizer
    )

    loss_fn = nn.MSELoss()

    train_and_compare_models(
        num_epochs=num_epochs, 
        batch_size=batch_size, 
        shape=shape, 
        loss_fn=loss_fn, 
        tt_model=tt_model,
        tt_optimizer=tt_optimizer, 
        golden_model=golden_model, 
        golden_optimizer=golden_optimizer
    )


@pytest.mark.parametrize(
    "shape",
    [
        (784, 10),
        (33, 127),
        (128, 20),
    ],
)
@pytest.mark.parametrize(
    "betas",
    [
        (0.9, 0.999),
        (0.8, 0.99),
    ],
)
@pytest.mark.parametrize(
    "weight_decay",
    [
        0.0,
        0.1,
    ],
)
def test_adam(shape, betas, weight_decay):
    torch.manual_seed(0)
    num_epochs = 10
    # Large learning rate to propagate possible errors faster
    learning_rate = 1
    batch_size = 64
    eps = 1e-8

    framework_model = MNISTLinear(input_size=shape[0], output_size=shape[1], bias=False)
    golden_model = MNISTLinear(input_size=shape[0], output_size=shape[1], bias=False)
    
    copy_model_params(framework_model, golden_model)

    tt_optimizer = forge.optimizers.Adam(learning_rate=learning_rate, epsilon=eps, beta1=betas[0], beta2=betas[1], weight_decay=weight_decay, bias_correction=True)
    golden_optimizer = torch.optim.Adam(golden_model.parameters(), lr=learning_rate, eps=eps, betas=betas, weight_decay=weight_decay)

    tt_model = forge.compile(
        framework_model, 
        sample_inputs=[torch.randn(batch_size, shape[0])],
        optimizer=tt_optimizer
    )

    loss_fn = nn.MSELoss()

    train_and_compare_models(
        num_epochs=num_epochs, 
        batch_size=batch_size, 
        shape=shape, 
        loss_fn=loss_fn, 
        tt_model=tt_model,
        tt_optimizer=tt_optimizer, 
        golden_model=golden_model, 
        golden_optimizer=golden_optimizer
    )


@pytest.mark.parametrize(
    "shape",
    [
        (784, 10),
        (33, 127),
        (128, 20),
    ],
)
@pytest.mark.parametrize(
    "alpha",
    [
        0.9,
        0.99,
    ],
)
@pytest.mark.parametrize(
    "momentum",
    [
        0.0,
        0.9,
    ],
)
@pytest.mark.parametrize(
    "weight_decay",
    [
        0.0,
        0.1,
    ],
)
@pytest.mark.parametrize(
    "centered",
    [
        False,
        True,
    ],
)
def test_rmsprop(shape, alpha, momentum, weight_decay, centered):
    torch.manual_seed(0)
    num_epochs = 10
    # Large learning rate to propagate possible errors faster
    learning_rate = 1
    batch_size = 64
    eps = 1e-8

    framework_model = MNISTLinear(input_size=shape[0], output_size=shape[1], bias=False)
    golden_model = MNISTLinear(input_size=shape[0], output_size=shape[1], bias=False)
    
    copy_model_params(framework_model, golden_model)

    tt_optimizer = forge.optimizers.RMSprop(learning_rate=learning_rate, alpha=alpha, momentum=momentum, epsilon=eps, weight_decay=weight_decay, centered=centered)
    golden_optimizer = torch.optim.RMSprop(golden_model.parameters(), lr=learning_rate, alpha=alpha, momentum=momentum, eps=eps, weight_decay=weight_decay, centered=centered)

    tt_model = forge.compile(
        framework_model, 
        sample_inputs=[torch.randn(batch_size, shape[0])],
        optimizer=tt_optimizer
    )

    loss_fn = nn.MSELoss()

    train_and_compare_models(
        num_epochs=num_epochs, 
        batch_size=batch_size, 
        shape=shape, 
        loss_fn=loss_fn, 
        tt_model=tt_model,
        tt_optimizer=tt_optimizer, 
        golden_model=golden_model, 
        golden_optimizer=golden_optimizer
    )
