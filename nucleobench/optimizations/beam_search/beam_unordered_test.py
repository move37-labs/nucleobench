"""Tests for beam_unordered.py

To test:
```zsh
pytest nucleobench/optimizations/beam_search/beam_unordered_test.py
```
"""

import pytest

from nucleobench.common import testing_utils
from nucleobench.optimizations.beam_search import beam_utils
from nucleobench.optimizations.beam_search import beam_unordered


@pytest.mark.parametrize(
    'edit_location_algo,edit_proposal_algo', 
    [('all', 'all'),
     ('all', 'random'),
     ('random', 'all'),
    ])
def test_beamsearch_init(edit_location_algo, edit_proposal_algo):
    beamsearch = beam_unordered.UnorderedBeamSearch(
        model_fn=testing_utils.CountLetterModel(
            vocab_i=1, flip_sign=True),  # Count 'C'
        start_sequence='ACTG',
        beam_size=10,
        edit_location_algo=edit_location_algo,
        edit_proposal_algo=edit_proposal_algo,
        random_n_loc=2,
    )
    assert list(beamsearch.beam.get_items()) == ['ACTG']


@pytest.mark.parametrize(
    'edit_location_algo,edit_proposal_algo', 
    [('all', 'all'),
     ('all', 'random'),
     ('random', 'all'),
    ])
def test_beam_run_correctness(edit_location_algo, edit_proposal_algo):
    
    beamsearch = beam_unordered.UnorderedBeamSearch(
        model_fn=testing_utils.CountLetterModel(
            vocab_i=1, flip_sign=True),  # Count 'C'
        start_sequence='ACTG',
        beam_size=5,
        edit_location_algo=edit_location_algo,
        edit_proposal_algo=edit_proposal_algo,
        random_n_loc=2,
        rng_seed=0,
    )
    assert list(beamsearch.beam.get_items()) == ['ACTG']
    
    beamsearch.run(n_steps=4)
    samples = beamsearch.get_samples(5)
    if edit_location_algo == 'all' and edit_proposal_algo == 'all':
        assert samples[0].count('C') == 4
    for s in samples:
        assert s.count('C') >= 1
        

@pytest.mark.parametrize(
    'edit_location_algo', 
    ['random', 'all'])
def test_locations_to_mutate(edit_location_algo):
    beamsearch = beam_unordered.UnorderedBeamSearch(
        model_fn=testing_utils.CountLetterModel(
            vocab_i=1, flip_sign=True),  # Count 'C'
        start_sequence='AAAAAA',
        positions_to_mutate=[1, 2],
        beam_size=5,
        edit_location_algo=edit_location_algo,
        edit_proposal_algo='all',
    )
    assert list(beamsearch.beam.get_items()) == ['AAAAAA']
    
    for _ in range(10):
        beamsearch.run(n_steps=4)
        samples = beamsearch.get_samples(5)
        for sample in samples:
            for idx in [0, 3, 4]:
                assert sample[idx] == 'A'  # unchanged.
            
    
    