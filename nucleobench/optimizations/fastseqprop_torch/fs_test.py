"""Tests for fs.py.

To test:
```zsh
pytest nucleobench/optimizations/fastseqprop_torch/fs_test.py
```
"""

import numpy as np
import torch

from nucleobench.common import string_utils
from nucleobench.common import testing_utils

from nucleobench.optimizations.fastseqprop_torch import fs


def test_init_sanity():
    init_args = fs.FastSeqProp.debug_init_args()
    fs.FastSeqProp(**init_args)


def test_fs_opt_param_init():
    init_args = fs.FastSeqProp.debug_init_args()
    init_args['start_sequence'] = 'AAAA'
    fs_opt = fs.FastSeqProp(**init_args)
    actual = fs_opt.opt_module.params.detach().numpy().squeeze(0)
    expected = string_utils.dna2tensor('AAAA').numpy()
    assert np.all(np.array_equal(actual, expected)), (actual, expected)

    fs_opt.reset('ACTGC')
    actual = fs_opt.opt_module.params.detach().numpy().squeeze(0)
    expected = string_utils.dna2tensor('ACTGC').numpy()
    assert np.all(np.array_equal(actual, expected)), (actual, expected)

def test_reset_sanity():
    init_args = fs.FastSeqProp.debug_init_args()
    init_args['start_sequence'] = 'AAAA'
    fs_opt = fs.FastSeqProp(**init_args)
    assert fs_opt.start_sequence == 'AAAA'
    _ = fs_opt.get_samples(1)[0]

def test_opt_changes_param():
    init_args = fs.FastSeqProp.debug_init_args()
    fs_opt = fs.FastSeqProp(**init_args)

    start_params = fs_opt.opt_module.params.detach().clone().numpy()
    fs_opt.run(n_steps=1)
    end_params = fs_opt.opt_module.params.detach().numpy()

    assert np.any(np.not_equal(start_params, end_params))


def test_correctness():
    torch.manual_seed(10)

    class ToOptimize(testing_utils.CountLetterModel):

        def inference_on_tensor(self, x):
            return -1 * super().inference_on_tensor(x)

    init_args = fs.FastSeqProp.debug_init_args()
    init_args['model_fn'] = ToOptimize()
    fs_opt = fs.FastSeqProp(**init_args)

    start_params = fs_opt.opt_module.params.detach().clone().numpy()
    start_energy = fs_opt.energy(batch_size=8).detach().clone().numpy().mean()

    energies = fs_opt.run(n_steps=10)

    final_params = fs_opt.opt_module.params.detach().numpy()
    final_energy = energies[-1].mean()

    assert np.any(np.not_equal(start_params, final_params)), final_params
    assert final_energy < start_energy
