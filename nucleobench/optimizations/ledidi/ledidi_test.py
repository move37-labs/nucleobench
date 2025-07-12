"""Test Ledidi.

To test:
```zsh
pytest nucleobench/optimizations/ledidi/ledidi_test.py
```
"""

import pytest

import numpy as np
import torch

from nucleobench.common import testing_utils
from nucleobench.optimizations.ledidi import ledidi

@pytest.mark.parametrize('positions_to_mutate', [False, True])
def test_init_sanity(positions_to_mutate):
    positions_to_mutate = [0] if positions_to_mutate else None
    ledidi.Ledidi(
        model_fn=testing_utils.CountLetterModel(),
        start_sequence='AA',
        positions_to_mutate=positions_to_mutate,
        train_batch_size=4,
    )


def test_get_samples():
    ld_opt = ledidi.Ledidi(
        model_fn=testing_utils.CountLetterModel(),
        start_sequence='AA',
        train_batch_size=4,
    )
    
    for num_samples in [2, 3]:
        ret = ld_opt.get_samples(num_samples)
        assert len(ret) == num_samples
        for st in ret:
            assert st == 'AA'
        

@pytest.mark.parametrize('positions_to_mutate', [False, True])
def test_correctness(positions_to_mutate):
    positions_to_mutate = [0] if positions_to_mutate else None

    # Counts 'C'
    model_fn = testing_utils.CountLetterModel(flip_sign=True, vocab_i=1)
    led_opt = ledidi.Ledidi(
        model_fn=model_fn,
        start_sequence='AA',
        train_batch_size=4,
        positions_to_mutate=positions_to_mutate,
        lr=1.0,
        rng_seed=10,
        )

    start_params = led_opt.designer.weights.detach().clone().numpy()
    start_energy = model_fn.inference_on_strings(['AA'])[0]

    energies = led_opt.run(n_steps=10)

    final_params = led_opt.designer.weights.detach().numpy()
    final_energy = energies[-1]

    assert np.any(np.not_equal(start_params, final_params)), final_params
    assert final_energy < start_energy
    
    ret = led_opt.get_samples(2)
    best_cnt = 0
    for st in ret:
        assert st.count('C') > 0, (st, final_energy, start_energy)
        best_cnt = max(best_cnt, st.count('C'))
    
    if positions_to_mutate:
        assert best_cnt == 1
    else:
        assert best_cnt == 2