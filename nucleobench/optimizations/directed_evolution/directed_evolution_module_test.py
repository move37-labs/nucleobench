"""Test the directed evolution module.

To test:
```zsh
pytest nucleobench/optimizations/directed_evolution/directed_evolution_module_test.py
```
"""

import pytest
import numpy as np

from nucleobench.common import testing_utils

from nucleobench.optimizations.directed_evolution import directed_evolution_module as de_mod

def _edit_distance(str1, str2) -> int:
    cnt = 0
    for c1, c2 in zip(str1, str2):
        if c1 != c2:
            cnt += 1
    return cnt

@pytest.mark.parametrize("use_must_change_mask", [True, False])
def test_single_bp_ism(use_must_change_mask: bool):
    base_seq = 'AAATG'
    if use_must_change_mask:
        must_change_mask = [True, False, True, True, False]
    else:
        must_change_mask = None
    isms = de_mod.single_bp_ism(
        base_seq=base_seq,
        positions=[0, 1, 2, 3, 4],
        vocab=['A', 'C', 'G', 'T'],
        must_change_mask=must_change_mask,
    )
    assert len(isms) == 5
    
    must_change_mask = must_change_mask or [False] * len(isms)
    for ism, must_change in zip(isms, must_change_mask):
        if must_change:
            assert _edit_distance(ism, base_seq) == 1
        else:
            assert _edit_distance(ism, base_seq) <= 1
        
        
def test_single_bp_ism_position():
    base_seq = 'AAATG'
    isms = de_mod.single_bp_ism(
        base_seq=base_seq,
        positions=[4],
        vocab=['A', 'C', 'G', 'T'],
    )
    assert len(isms) == 1
    
    for ism in isms:
        assert ism[:4] == 'AAAT'
        assert _edit_distance(ism, base_seq) <= 1
        
        
def test_batchify():
    lst = [1, 2, 3, 4, 5]
    assert de_mod.batchify(lst, 2) == [[1, 2], [3, 4], [5]]
    assert de_mod.batchify(lst, 3) == [[1, 2, 3], [4, 5]]
    assert de_mod.batchify(lst, 4) == [[1, 2, 3, 4], [5]]
    assert de_mod.batchify(lst, 5) == [[1, 2, 3, 4, 5]]
    
    
def test_get_predictions():
    ret = de_mod.get_predictions(
        testing_utils.CountLetterModel(),
        ['AA', 'CC', 'GG', 'TC'],
        batch_size=2)
    assert np.all(ret == [0, 2, 0, 1])
    

def test_single_step():
    _, best_score = de_mod.single_step(
        cur_seqs=['AA', 'GG', 'GG', 'TC'],
        model=testing_utils.CountLetterModel(flip_sign=True),
        batch_size=2)
    assert best_score <= -1
    

@pytest.mark.parametrize('use_tism,location_only,pos_to_mutate', [
    (False, True, False),
    (True, True, False),
    (True, False, False),
    (False, True, True),
    (True, True, True),
    (True, False, True),
])
def test_evolve_sanity(use_tism,location_only,pos_to_mutate):
    if use_tism:
        tism_args = de_mod.TISMArgs(
            location_only=location_only,
            budget=2,
            fraction_tism=0.5)
    else:
        tism_args = None
    best_seqs, best_score, _ = de_mod.evolve(
        model=testing_utils.CountLetterModel(flip_sign=True),
        seqs=['AA'*5, 'GG'*5, 'GG'*5, 'TC'*5],
        max_iter=2,
        batch_size=2,
        tism_args=tism_args,
        positions=None if pos_to_mutate is False else [0, 1, 2, 3, 4],
    )
    assert best_score <= -1
    for s in best_seqs:
        assert s.count('C') == -1 * best_score
        
        
def test_positions_from_tism_sanity():
    de_mod.positions_from_tism(
        base_seq='AAATG',
        model=testing_utils.CountLetterModel(),
        positions=[0, 1, 2, 3, 4],
        tism_args=de_mod.TISMArgs(
            location_only=True,
            fraction_tism=0.5, 
            budget=5),
    )