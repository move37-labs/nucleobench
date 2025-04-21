"""Tests for adalead_ref.py

To test:
```zsh
pytest nucleobench/optimizations/ada/adalead/adalead_v2_test.py
```
"""

import numpy as np

from nucleobench.optimizations.ada.adabeam.adabeam import AdaBeam
from nucleobench.common import testing_utils


def test_adalead_convergence():
    model = testing_utils.CountLetterModel(
        flip_sign=True,
    )

    start_seq = "AAAAAA"
    start_score = model([start_seq])[0]
    assert start_score == 0

    beam_size = 20
    adalead = AdaBeam(
        model_fn=model,
        seed_sequence=start_seq,
        beam_size=beam_size,
        mutations_per_sequence=2,
        threshold=0.25,
        n_rollouts_per_root=4,
        eval_batch_size=1,
        rng_seed=42,
    )

    adalead.run(n_steps=20)

    out_seqs = adalead.get_samples(beam_size)
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

    beam_size = 2
    adalead = AdaBeam(
        model_fn=model,
        seed_sequence=start_seq,
        positions_to_mutate=[0, 1],
        beam_size=beam_size,
        mutations_per_sequence=1,
        threshold=0.4,
        n_rollouts_per_root=4,
        eval_batch_size=1,
        rng_seed=42,
    )

    for _ in range(4):
        adalead.run(n_steps=1)

        out_seqs = adalead.get_samples(beam_size)
        for seq in out_seqs:
            for s in seq[2:]:
                assert s == 'A', seq
