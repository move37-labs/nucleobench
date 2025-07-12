"""Tests for beam_ordered.py

To test:
```zsh
pytest nucleobench/optimizations/beam_search/beam_ordered_test.py
```
"""

import pytest

from nucleobench.common import testing_utils
from nucleobench.optimizations.beam_search import beam_utils
from nucleobench.optimizations.beam_search import beam_ordered as beam


@pytest.mark.parametrize('init_order_method', ['sequential', 'tism_fixed', 'tism_reverse'])
def test_beamsearch_init(init_order_method):
    beamsearch = beam.OrderedBeamSearch(
        model_fn=testing_utils.CountLetterModel(
            vocab_i=1, flip_sign=True),  # Count 'C'
        start_sequence='ACTG',
        beam_size=10,
        init_order_method=init_order_method,
    )
    assert list(beamsearch.beam.get_items()) == ['ACTG']
    if init_order_method == 'tism_fixed':
        # Changing 'C' should be the last thing to do.
        assert beamsearch._search_order[-1] == 1
        
def test_get_potential_moves():
    beam_obj = beam_utils.Beam(5)
    beam_obj.put([(2, 'AA'), (1, 'AT')])
    pot1 = beam.get_potential_moves(beam_obj, 0, ['A', 'C', 'G', 'T'])
    assert set(pot1) == {'AA', 'CA' ,'GA', 'TA',
                         'AT', 'CT', 'GT', 'TT'}
    
    pot2 = beam.get_potential_moves(beam_obj, 1, ['A', 'C', 'G', 'T'])
    assert set(pot2) == {'AA', 'AC' ,'AT', 'AG'}
    

@pytest.mark.parametrize('init_order_method,', ['sequential', 'tism_fixed', 'tism_reverse'])
def test_beam_run_sanity(init_order_method):
    beamsearch = beam.OrderedBeamSearch(
        model_fn=testing_utils.CountLetterModel(
            vocab_i=1, flip_sign=True),  # Count 'C'
        start_sequence='ACTG',
        beam_size=5,
        init_order_method=init_order_method,
    )
    assert list(beamsearch.beam.get_items()) == ['ACTG']
    
    beamsearch.run(n_steps=2)
    samples = beamsearch.get_samples(5)
    for s in samples:
        assert s.count('C') >= 1
    
    beamsearch.run(n_steps=2)
    samples = beamsearch.get_samples(5)
    assert samples[0].count('C') == 4
    for s in samples:
        assert s.count('C') >= 1
        
        
def test_maxqueue_correctness():
    beamsearch = beam.OrderedBeamSearch(
        model_fn=testing_utils.CountLetterModel(
            vocab_i=1, flip_sign=True),  # Count 'C'
        start_sequence='ACTG',
        beam_size=5,
        init_order_method='sequential',
        use_priority_queue=True,
        priority_queue_size=1,
    )
    beamsearch.run(n_steps=4)
    samples = beamsearch.get_samples(1)[0]
    assert samples.count('C') == 4
    
    
def test_positions_to_mutate():
    beamsearch = beam.OrderedBeamSearch(
        model_fn=testing_utils.CountLetterModel(
            vocab_i=1, flip_sign=True),  # Count 'C'
        start_sequence='AAAAA',
        positions_to_mutate=[1, 2],
        beam_size=5,
        init_order_method='sequential',
    )
    for _ in range(4):
        beamsearch.run(n_steps=4)
        samples = beamsearch.get_samples(1)[0]
        for idx in [0, 3, 4]:
            assert samples[idx] == 'A'  # unchanged.
            
def test_random_seed():
    search_orders = []
    args = dict(
        model_fn=testing_utils.CountLetterModel(),
        start_sequence='AAAAA',
        beam_size=5,
        init_order_method='random',
    )
    for rng_seed in range(4):
        beamsearch1 = beam.OrderedBeamSearch(
            **args, rng_seed=rng_seed)
        beamsearch2 = beam.OrderedBeamSearch(
            **args, rng_seed=rng_seed)
        assert beamsearch1._search_order == beamsearch2._search_order
        search_orders.append(tuple(beamsearch1._search_order))
    assert len(set(search_orders)) != 1
    
    
def test_get_samples_fail():
    beamsearch = beam.OrderedBeamSearch(
        model_fn=testing_utils.CountLetterModel(
            vocab_i=1, flip_sign=True),  # Count 'C'
        start_sequence='ACTG',
        beam_size=5,
        init_order_method='random',
    )
    assert list(beamsearch.beam.get_items()) == ['ACTG']
    
    beamsearch.run(n_steps=2)
    with pytest.raises(ValueError):
        beamsearch.get_samples(6)