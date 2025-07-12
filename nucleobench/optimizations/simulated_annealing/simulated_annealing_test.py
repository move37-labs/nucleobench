import numpy as np

from nucleobench.optimizations.simulated_annealing.simulated_annealing import (
    SimulatedAnnealing,
)
from nucleobench.common import testing_utils


def test_simulated_annealing():
    model = testing_utils.CountLetterModel(
        flip_sign=True,
    )

    start_seq = "AAAAAA"
    start_score = model([start_seq])[0]
    assert start_score == 0

    # Try editing all positions.
    sa = SimulatedAnnealing(
        model_fn=model,
        start_sequence=start_seq,
        positions_to_mutate=None,
        polynomial_decay_a=1.0,
        polynomial_decay_b=1.0,
        polynomial_decay_p=1.0,
        n_mutations_per_proposal=1,
        rng_seed=42,
    )
    sa.run(n_steps=100)

    out_seq = sa.get_samples(1)[0]
    out_seq_score = model([out_seq])[0]
    # print(out_seq, out_seq_score)

    assert out_seq_score < start_score

    # Try not editing the last position.
    sa = SimulatedAnnealing(
        model_fn=model,
        start_sequence=start_seq,
        positions_to_mutate=list(range(len(start_seq) - 1)),
        polynomial_decay_a=1.0,
        polynomial_decay_b=1.0,
        polynomial_decay_p=1.0,
        n_mutations_per_proposal=1,
        rng_seed=42,
    )
    sa.run(n_steps=100)

    out_seq = sa.get_samples(1)[0]
    out_seq_score = model([out_seq])[0]
    # print(out_seq, out_seq_score)

    assert out_seq_score < start_score


if __name__ == "__main__":
    test_simulated_annealing()
