"""Tests for adalead_ref.py

To test:
```zsh
pytest nucleobench/optimizations/ada/adalead/adalead_ref_test.py
```
"""

import pytest
import numpy as np

from nucleobench.optimizations.ada.adalead.adalead_ref import AdaLeadRef
from nucleobench.common import testing_utils


def test_adalead_ref():
    model = testing_utils.CountLetterModel(
        flip_sign=True,
    )

    start_seq = "AAAAAA"
    start_score = model([start_seq])[0]
    assert start_score == 0

    seq_batch_size = 20
    adalead = AdaLeadRef(
        model_fn=model,
        start_sequence=start_seq,
        sequences_batch_size=seq_batch_size,
        model_queries_per_batch=100,
        threshold=0.25,
        rho=2,
        eval_batch_size=1,
        rng_seed=42,
        mutation_rate=0.25,
        recombination_rate=0.2,
    )

    adalead.run(n_steps=20)

    out_seqs = adalead.get_samples(seq_batch_size)
    out_seq_scores = np.array([model([s])[0] for s in out_seqs])

    assert out_seq_scores[0] < start_score


def test_positions_to_mutate():
    """No matter how many iterations, positions outside `positions_to_mutate` shouldn't change."""
    model = testing_utils.CountLetterModel(
        flip_sign=True,
    )

    start_seq = "A" * 100
    start_score = model([start_seq])[0]
    assert start_score == 0

    seq_batch_size = 2
    adalead = AdaLeadRef(
        model_fn=model,
        start_sequence=start_seq,
        positions_to_mutate=[0, 1],
        sequences_batch_size=seq_batch_size,
        model_queries_per_batch=11,
        mutation_rate=1.0,
        recombination_rate=0.2,
        threshold=0.4,
        rho=2,
        eval_batch_size=1,
        rng_seed=42,
    )

    for i in range(4):
        adalead.run(n_steps=1)
        print(f'Finished step {i}')

        out_seqs = adalead.get_samples(seq_batch_size)
        for seq in out_seqs:
            for s in seq[2:]:
                assert s == 'A', seq


def test_zero_is_not_none():
    AdaLeadRef(
        model_fn=testing_utils.CountLetterModel(),
        start_sequence="A" * 100,
        sequences_batch_size=2,
        model_queries_per_batch=11,
        mutation_rate=0.1,
        recombination_rate=0.0,
        threshold=0.4,
        rho=2,
        eval_batch_size=1,
        rng_seed=42,
    )