"""Tests for fs.py.

To test:
```zsh
pytest nucleobench/optimizations/fastseqprop_torch/fs_test.py
```
"""

import numpy as np
import torch

from nucleobench.common import testing_utils

from . import fs


def test_init_sanity():
    init_args = fs.FastSeqProp.debug_init_args()
    fs.FastSeqProp(**init_args)


def test_reset_sanity():
    init_args = fs.FastSeqProp.debug_init_args()
    init_args["start_sequence"] = "AAAA"
    fs_opt = fs.FastSeqProp(**init_args)
    assert fs_opt.start_sequence == "AAAA"
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
    init_args = fs.FastSeqProp.debug_init_args()
    init_args["model_fn"] = testing_utils.CountLetterModel(flip_sign=True)
    fs_opt = fs.FastSeqProp(**init_args)

    start_params = fs_opt.opt_module.params.detach().clone().numpy()
    start_energy = fs_opt.energy(batch_size=8).detach().clone().numpy().mean()

    energies = fs_opt.run(n_steps=10)

    final_params = fs_opt.opt_module.params.detach().numpy()
    final_energy = energies[-1].mean()

    assert np.any(np.not_equal(start_params, final_params)), final_params
    assert final_energy < start_energy


def test_get_samples_first_entry_is_argmax():
    """get_samples should always return the deterministic argmax as samples[0]."""
    torch.manual_seed(0)
    init_args = fs.FastSeqProp.debug_init_args()
    init_args["start_sequence"] = "ACGT"
    fs_opt = fs.FastSeqProp(**init_args)

    # Run a few steps so the PWM has moved away from the initial state.
    fs_opt.run(n_steps=5)

    best = fs_opt.get_best_sequence()

    # Check for n_samples=1: only the argmax is returned.
    samples_1 = fs_opt.get_samples(n_samples=1)
    assert len(samples_1) == 1
    assert samples_1[0] == best

    # Check for n_samples=N: first entry is argmax, total length is correct.
    for n in [2, 5]:
        samples = fs_opt.get_samples(n_samples=n)
        assert len(samples) == n
        assert samples[0] == best, (
            f"samples[0]={samples[0]!r} != best={best!r} for n_samples={n}"
        )

    # Check that get_best_sequence is deterministic across calls.
    assert fs_opt.get_best_sequence() == best


def test_respects_pos_to_mutate():
    start_sequence = "A" * 20
    positions_to_mutate = [2, 5, 10]

    model_fn = testing_utils.CountLetterModel()

    opt = fs.FastSeqProp(
        model_fn=model_fn,
        start_sequence=start_sequence,
        positions_to_mutate=positions_to_mutate,
        learning_rate=0.1,
        batch_size=4,
        eta_min=1e-6,
    )

    for _ in range(10):
        opt.run(n_steps=1)
        proposals = opt.get_samples(n_samples=10)
        for proposal in proposals:
            testing_utils.assert_proposal_respects_positions_to_mutate(
                start_sequence, proposal, positions_to_mutate
            )
