"""Tests for adabeam.py

To test:
```zsh
pytest nucleobench/optimizations/ada/adabeam/adabeam_test.py
```
"""

from itertools import product
import pytest
import numpy as np
import random

from nucleobench.optimizations.ada.adabeam.adabeam import AdaBeam
from nucleobench.common import testing_utils
    

@pytest.mark.parametrize(
    'skip_repeat_sequences', 
    product(
        [True, False],  # skip_repeat_sequences
    ))
def test_adabeam_sanity(skip_repeat_sequences):
    model_fn = testing_utils.CountLetterModel(flip_sign=True)

    start_seq = 'A' * 100
    start_score = model_fn([start_seq])[0]
    assert start_score == 0

    beam_size = 20
    kwargs = AdaBeam.debug_init_args()
    kwargs['model_fn'] = model_fn
    kwargs['start_sequence'] = start_seq
    kwargs['beam_size'] = beam_size
    kwargs['skip_repeat_sequences'] = skip_repeat_sequences
    adabeam = AdaBeam(**kwargs)

    adabeam.run(n_steps=2)

    out_seqs = adabeam.get_samples(beam_size)
    del out_seqs


def test_adabeam_convergence():
    model_fn = testing_utils.CountLetterModel(flip_sign=True)

    start_seq = 'A' * 100
    start_score = model_fn([start_seq])[0]
    assert start_score == 0

    kwargs = AdaBeam.debug_init_args()
    kwargs['model_fn'] = model_fn
    kwargs['start_sequence'] = start_seq
    kwargs['skip_repeat_sequences'] = True
    adabeam = AdaBeam(**kwargs)

    adabeam.run(n_steps=2)

    out_seqs = adabeam.get_samples(kwargs['beam_size'])
    out_seq_scores = np.array([model_fn([s])[0] for s in out_seqs])

    assert out_seq_scores[0] < start_score


def test_positions_to_mutate():
    """No matter how many iterations, positions outside `positions_to_mutate` shouldn't change."""
    model_fn = testing_utils.CountLetterModel(flip_sign=True)

    start_seq = "A" * 100
    start_score = model_fn([start_seq])[0]
    assert start_score == 0

    beam_size = 2
    kwargs = AdaBeam.debug_init_args()
    kwargs['model_fn'] = model_fn
    kwargs['start_sequence'] = start_seq
    kwargs['skip_repeat_sequences'] = True
    kwargs['beam_size'] = beam_size
    adabeam = AdaBeam(**kwargs, positions_to_mutate=list(range(20)))

    for _ in range(4):
        adabeam.run(n_steps=1)

        out_seqs = adabeam.get_samples(beam_size)
        for seq in out_seqs:
            for s in seq[20:]:
                assert s == 'A', seq
                
                
@pytest.mark.parametrize('eval_batch_size', [1, 2, 4])
def test_eval_batch_size_sanity(eval_batch_size):
    """Test that `eval_batch_size` works."""
    model_fn = testing_utils.CountLetterModel(flip_sign=True)

    start_seq = "A" * 100
    start_score = model_fn([start_seq])[0]
    assert start_score == 0

    kwargs = AdaBeam.debug_init_args()
    kwargs['model_fn'] = model_fn
    kwargs['start_sequence'] = start_seq
    kwargs['eval_batch_size'] = eval_batch_size
    adabeam = AdaBeam(**kwargs)

    for _ in range(4):
        adabeam.run(n_steps=4)

        # TODO(joelshor):
        # Add correctness checks.
        
        
def test_eval_batch_size_consistency():
    """Test that `eval_batch_size` is consistent."""
    model_fn = testing_utils.CountLetterModel(flip_sign=True)

    seqs = [''.join(random.choices(['A', 'G', 'T' ,'C'], k=100)) for _ in range(10)]

    kwargs = AdaBeam.debug_init_args()
    kwargs['model_fn'] = model_fn
    kwargs['start_sequence'] = 'A' * 100
    
    kwargs['eval_batch_size'] = 1
    adabeam1 = AdaBeam(**kwargs)
    
    kwargs['eval_batch_size'] = 2
    adabeam2 = AdaBeam(**kwargs)
    
    kwargs['eval_batch_size'] = 4
    adabeam4 = AdaBeam(**kwargs)
    
    scores1 = adabeam1.get_batched_fitness(seqs)
    scores2 = adabeam2.get_batched_fitness(seqs)
    scores4 = adabeam4.get_batched_fitness(seqs)
    
    assert np.array_equal(scores1, scores2)
    assert np.array_equal(scores1, scores4)
    assert np.array_equal(scores2, scores4)