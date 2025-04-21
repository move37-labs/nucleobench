"""Test attribution_lib.py.

To test:
pytest nucleobench/common/attribution_lib_torch_test.py
"""

import pytest
import numpy as np
import torch
from torch import nn

from nucleobench.common import attribution_lib_torch
from nucleobench.common import testing_utils

class TestNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 3),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

class Nonlinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class FixedNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_array = np.array([1.0, 2.0])
        self.linear_layer = torch.tensor([self.layer_array], dtype=torch.float).t()

    def forward(self, x):
        logits = torch.matmul(x, self.linear_layer)
        return logits

@pytest.mark.parametrize('times', [1, 3, 5])
def test_expected_gradient(times):
    """Regardless of number of times, noisy grads should be the same for a linear model."""
    input_tensor = torch.randn(2)
    model = FixedNN()

    grads = attribution_lib_torch.noisy_grads_torch(
        input_tensor=input_tensor,
        model=model,
        noise_stdev=1.0,
        times=times)
    
    assert grads.shape == (times, 2)
    # For linear models, grads should all be the linear layer.
    for i in range(times):
        assert np.array_equal(grads[i, ...], model.layer_array), (
            grads[i, ...], model.layer_array, grads)
        
def test_callable():
    """Check that additional arguments are used."""
    input_tensor = torch.randn(2)
    model = FixedNN()

    def override_callable(x):
        return 2.0 * model(x)
    grads = attribution_lib_torch.noisy_grads_torch(
        input_tensor=input_tensor,
        model=override_callable,
        noise_stdev=1.0,
        times=3,
    )
    
    assert grads.shape == (3, 2)
    # For linear models, grads should all be the linear layer.
    for i in range(3):
        assert np.array_equal(grads[i, ...], [2.0, 4.0]), (
            grads[i, ...], [2.0, 4.0], grads)

def test_noisy_grads():
    input_tensor = torch.randn(10)
    model = TestNeuralNetwork()
    
    noisy_grads = attribution_lib_torch.noisy_grads_torch(
        input_tensor=input_tensor,
        model=model,
        noise_stdev=0.25,
        times=5)
    assert noisy_grads.shape == (5, 10)
    # For linear models, grads should all be the same.
    for i in range(5):
        assert np.array_equal(noisy_grads[i, ...], noisy_grads[0, ...])


def test_smoothgrad_torch():
    input_tensor = torch.randn(10)
    model = TestNeuralNetwork()
    
    grad = attribution_lib_torch.smoothgrad_torch(
        input_tensor=input_tensor,
        model=model,
        noise_stdev=0.25,
        times=5)

    assert grad.shape == (10,)


def test_noisy_grads_different():
    """In a nonlinear network, gradients should be different."""
    input_tensor = torch.randn(10)
    model = Nonlinear()
    
    noisy_grads = attribution_lib_torch.noisy_grads_torch(
        input_tensor=input_tensor,
        model=model,
        noise_stdev=1.0,
        times=5)
    assert noisy_grads.shape == (5, 10)
    # For linear models, grads should all be the same.
    for i in range(1, 5):
        assert not np.array_equal(noisy_grads[i, ...], noisy_grads[0, ...])
        
# TODO(joelshor): Add a unit test for nucleotide-specific TISM.
def test_smoothgrad_torch_idx_sanity():
    input_tensor = torch.randn(4, 5)
    model = testing_utils.CountLetterModel()
    
    grad = attribution_lib_torch.smoothgrad_torch(
        input_tensor=input_tensor,
        model=model.inference_on_tensor,
        noise_stdev=0.0,
        times=1)
    assert grad.shape == (4, 5)
    
    for idx in range(5):
        grad_singlebp = attribution_lib_torch.smoothgrad_torch(
            input_tensor=input_tensor,
            model=model.inference_on_tensor,
            noise_stdev=0.25,
            times=5,
            idxs=[idx])
        assert grad_singlebp.shape == (4, 1)
        assert np.all(grad[:, idx] == grad_singlebp[:, 0])
        
        
def test_apply_gradient_mask():
    """[idx] and idx should give the same result."""
    input_tensor = torch.randn(1, 10)
    model = TestNeuralNetwork()
    output_tensor = model(input_tensor)
    output_tensor = output_tensor.reshape(1, 1, 3)
    
    x1, x_grad = attribution_lib_torch.apply_gradient_mask(
        output_tensor, [0])
    assert x1.shape == (1, 1, 3)
    assert x_grad.shape == (1, 1, 1)
    
    x2, x_grad = attribution_lib_torch.apply_gradient_mask(
        output_tensor, [0, 1])
    assert x2.shape == (1, 1, 3)
    assert x_grad.shape == (1, 1, 2)
    
    assert (x1 == x2).all(), x1 == x2