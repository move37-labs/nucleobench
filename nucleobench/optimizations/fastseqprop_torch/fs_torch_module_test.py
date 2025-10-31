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
from nucleobench.common import testing_utils

from . import fs_torch_module as fst

def test_shape_sanity():
    start_tensor = string_utils.dna2tensor('ACTGC')
    fs_opt = fst.TorchFastSeqPropOptimizer(
        start_logits=torch.unsqueeze(start_tensor, dim=0),
        positions_to_mutate=[0, 2, 3],
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
    
    fs_opt = fst.TorchFastSeqPropOptimizer(
        start_logits=string_utils.dna2tensor_batch([seed], vocab_list=vocab),
        positions_to_mutate=[0, 2, 3],
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
    fs_opt = fst.TorchFastSeqPropOptimizer(
        start_logits=torch.unsqueeze(start_tensor, dim=0),
        positions_to_mutate=[0, 2],
        vocab_len=4,
    )
    all_params = list(fs_opt.parameters())
    assert len(all_params) == 1
    param = all_params[0]
    assert list(param.shape) == [1, 4, len(start_str)]
    assert param.requires_grad == True
    
    
def test_respects_pos_to_mutate():
    # Make a random 20-long sequence.
    start_sequence = ''.join(np.random.choice(list('ACGT'), size=20))
    positions_to_mutate = [2, 5, 10]
    start_probs = string_utils.dna2tensor(start_sequence)
    start_probs = torch.unsqueeze(start_probs, 0)
    
    module = fst.TorchFastSeqPropOptimizer(
        start_logits=start_probs,
        positions_to_mutate=positions_to_mutate
    )

    optimizer = torch.optim.Adam(module.parameters(), lr=0.1)

    for _ in range(10):
        samples = module.get_samples_onehot(n_samples=4)
        # Check that all samples are unchanged outside of positions to mutate.
        proposals = string_utils.tensor2dna_batch(samples.detach().numpy())
        for proposal in proposals:
            testing_utils.assert_proposal_respects_positions_to_mutate(
                start_sequence, proposal, positions_to_mutate)
        
        # Create a dummy loss
        optimizer.zero_grad()
        loss = samples.sum()
        loss.backward()
        optimizer.step()

    samples_onehot = module.get_samples_onehot(n_samples=10)
    proposals = string_utils.tensor2dna_batch(samples_onehot.detach().numpy())

    for proposal in proposals:
        testing_utils.assert_proposal_respects_positions_to_mutate(
            start_sequence, proposal, positions_to_mutate)