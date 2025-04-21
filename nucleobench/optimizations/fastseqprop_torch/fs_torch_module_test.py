"""Tests for fs_torch_module.py.

To test:
```zsh
pytest nucleobench/optimizations/fastseqprop_torch/fs_torch_module_test.py
```
"""

import numpy as np
import pytest
import torch

from nucleobench.common import string_utils

from nucleobench.optimizations.fastseqprop_torch import fs_torch_module as fs_module

def test_shape_sanity():
    start_tensor = string_utils.dna2tensor('ACTGC')
    fs_opt = fs_module.TorchFastSeqPropOptimizer(
        torch.unsqueeze(start_tensor, dim=0),
        positions_to_mutate = [0, 2, 3],
        vocab_len=4,
    )
    
    probs = fs_opt.get_probs()
    assert probs.ndim == 3
    assert list(probs.shape) == [1, 4, 5]
    
    samples_onehot = fs_opt.get_samples_onehot(3)
    assert list(samples_onehot.shape) == [3, 4, 5]
    

def test_prob_correctness():
    seed = 'ACTGC'
    vocab = ['A', 'C', 'G', 'T']
    
    fs_opt = fs_module.TorchFastSeqPropOptimizer(
        string_utils.dna2tensor_batch([seed], vocab_list=vocab),
        positions_to_mutate = [0, 2, 3],
        vocab_len=4,
    )
    probs = fs_opt.get_probs().detach().numpy()
    probs = probs.squeeze(0)
    for prob_v, expected_char in zip(np.transpose(probs), seed):
        mll_char = vocab[np.argmax(prob_v)]
        assert mll_char == expected_char
    
@pytest.mark.parametrize("start_str", [
    'ACTGC',
    'ACTG',
    'ACT',
])
def test_params(start_str: str):
    start_tensor = string_utils.dna2tensor(start_str)
    fs_opt = fs_module.TorchFastSeqPropOptimizer(
        torch.unsqueeze(start_tensor, dim=0),
        positions_to_mutate = [0, 2],
        vocab_len=4,
    )
    all_params = list(fs_opt.parameters())
    assert len(all_params) == 1
    param = all_params[0]
    assert list(param.shape) == [1, 4, len(start_str)]
    assert param.requires_grad == True