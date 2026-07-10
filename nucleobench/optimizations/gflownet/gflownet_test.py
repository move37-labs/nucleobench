"""Tests for gflownet.py.

To test:
```zsh
pytest nucleobench/optimizations/gflownet/gflownet_test.py -v
```
"""

import numpy as np
import torch

from nucleobench.common import constants, testing_utils

from . import gflownet

VOCAB = constants.VOCAB


def test_init_sanity():
    """Construct a GFlowNet from debug_init_args without error."""
    init_args = gflownet.GFlowNet.debug_init_args()
    gflownet.GFlowNet(**init_args)


def test_registered():
    """GFlowNet is retrievable from the registry, constructable, runnable, and samples correctly."""
    from nucleobench import optimizations

    assert optimizations.get_optimization("gflownet") is gflownet.GFlowNet

    init_args = gflownet.GFlowNet.debug_init_args()
    opt = gflownet.GFlowNet(**init_args)
    opt.run(n_steps=1)

    L = len(init_args["start_sequence"])
    samples = opt.get_samples(2)

    assert len(samples) == 2
    for s in samples:
        assert len(s) == L, f"Expected length {L}, got {len(s)}: {s!r}"
        assert all(c in VOCAB for c in s), f"Sequence contains non-VOCAB chars: {s!r}"


def test_learns_to_minimize():
    """GFlowNet actually learns to minimize oracle energy on CountLetterModel.

    Oracle: CountLetterModel() with default vocab_i=1 → energy = count of 'C'.
    Minimising drives the model away from 'C', so trained samples should have
    fewer C's than an untrained (uniform) policy.

    Three checks:
    1. Baseline (untrained) mean energy is near 0.25 * L — catches a trivial
       sampler that always emits start_sequence or is otherwise degenerate.
    2. final_energy < baseline_energy by a clear margin — the learning check.
    3. Mirror guard: with flip_sign=True (energy = -count_C), minimising should
       *increase* C count. This catches an incorrect reward sign implementation.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    L = 12
    batch_size = 32
    n_train_steps = 250

    oracle = testing_utils.CountLetterModel()

    opt = gflownet.GFlowNet(
        model_fn=oracle,
        start_sequence="A" * L,
        beta=2.0,
        learning_rate=1e-3,
        batch_size=batch_size,
        hidden_dim=64,
        rnd_seed=42,
    )

    # Measure baseline energy (untrained policy should be roughly uniform → ~0.25*L).
    baseline_seqs = opt.get_samples(batch_size)
    baseline_energies = np.array(oracle(baseline_seqs), dtype=np.float32)
    baseline_energy = float(baseline_energies.mean())

    expected_uniform = 0.25 * L  # = 3.0 for L=12
    assert 1.0 < baseline_energy < expected_uniform + 2.0, (
        f"Baseline energy {baseline_energy:.2f} is far from expected ~{expected_uniform:.1f}; "
        "the untrained policy may be degenerate."
    )

    # Train.
    opt.run(n_steps=n_train_steps)

    # Measure final energy.
    final_seqs = opt.get_samples(batch_size)
    final_energies = np.array(oracle(final_seqs), dtype=np.float32)
    final_energy = float(final_energies.mean())

    # The trained policy should be meaningfully lower-energy than baseline.
    margin = 1.0  # at least 1 count of 'C' fewer on average
    assert final_energy < baseline_energy - margin, (
        f"GFlowNet did not learn to minimise: "
        f"final_energy={final_energy:.2f}, baseline={baseline_energy:.2f}, "
        f"required margin={margin}"
    )

    # Secondary: mean C-count in trained samples is below untrained baseline.
    baseline_c_count = baseline_energy  # CountLetterModel energy == count of 'C'
    final_c_count = final_energy
    assert final_c_count < baseline_c_count, (
        f"Mean C-count did not decrease: "
        f"trained={final_c_count:.2f}, baseline={baseline_c_count:.2f}"
    )

    # ----------------------------------------------------------------
    # Mirror-image guard: flip_sign=True → energy = -count_C.
    # Minimising energy should now *increase* C count.
    # ----------------------------------------------------------------
    torch.manual_seed(42)
    np.random.seed(42)

    oracle_flip = testing_utils.CountLetterModel(flip_sign=True)

    opt_flip = gflownet.GFlowNet(
        model_fn=oracle_flip,
        start_sequence="A" * L,
        beta=2.0,
        learning_rate=1e-3,
        batch_size=batch_size,
        hidden_dim=64,
        rnd_seed=42,
    )

    baseline_flip_seqs = opt_flip.get_samples(batch_size)
    # Use the original (non-flipped) oracle to count actual C's.
    baseline_flip_c = float(np.array(oracle(baseline_flip_seqs)).mean())

    opt_flip.run(n_steps=n_train_steps)

    final_flip_seqs = opt_flip.get_samples(batch_size)
    final_flip_c = float(np.array(oracle(final_flip_seqs)).mean())

    assert final_flip_c > baseline_flip_c + margin, (
        f"Mirror guard failed: with flip_sign=True the GFlowNet should learn to "
        f"INCREASE C count. "
        f"final_c={final_flip_c:.2f}, baseline_c={baseline_flip_c:.2f}, "
        f"required margin={margin}"
    )
