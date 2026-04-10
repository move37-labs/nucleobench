"""Tests for adalead utils.

To test:
```zsh
pytest nucleobench/optimizations/ada/ada_utils_test.py
```
"""
# TODO(joelshor): Write test for `get_tism_edits_and_probs`.

import numpy as np
import pytest
import random

from nucleobench.common import testing_utils

from nucleobench.optimizations.ada import ada_utils


# (sequence length, mutation rate)
PARAMS_TO_TEST_ = [(10, .1),
                   (10, .5),
                   (10, .7),
                   (100, .1),
                   (10, .5),
                   (10, .7),
                  ]

LIKELIHOOD_FNS_ = [
    ada_utils.num_edits_likelihood_adalead_legacy,
    ada_utils.num_edits_likelihood_adabeam]


@pytest.mark.parametrize('likelihood_fn', LIKELIHOOD_FNS_)
def test_num_edits_likelihood_legacy_prob_dist(likelihood_fn):
    for sequence_length, mutation_rate in PARAMS_TO_TEST_:
        actual_sum = np.sum(likelihood_fn(np.arange(sequence_length+1), sequence_length, mutation_rate))
        expected_sum = 1.0
        np.testing.assert_allclose(actual_sum, expected_sum)


@pytest.mark.parametrize('sequence_length, mutation_rate', PARAMS_TO_TEST_)
def test_explicit_likelihood_legacy_equivalence(
    sequence_length, mutation_rate, num_samples=150000, atol=0.002):
    """Tests that adalead ref and explicit likelihood are equivalent."""
    rng = random.Random(0)
    alphabet = 'ACTG'
    num_changes = []
    for _ in range(num_samples):
        num_edits = 0
        while num_edits == 0:
            mutant = ada_utils.generate_random_mutant(
                sequence='A' * sequence_length,
                positions_to_mutate=list(range(sequence_length)),
                mu=mutation_rate,
                alphabet=alphabet,
                rng=rng
            )
            num_edits = sequence_length - mutant.count('A')
        num_changes.append(num_edits)
    
    actual, expected = [], []
    for num_edits in sorted(np.unique(num_changes)):
        actual.append(num_changes.count(num_edits) / float(len(num_changes)))
        expected.append(ada_utils.num_edits_likelihood_adalead_legacy(
            num_edits=np.array([num_edits]),
            seq_len=sequence_length,
            mu=mutation_rate
        )[0])
    np.testing.assert_allclose(
        actual, expected, atol=atol,
        err_msg=f'{num_edits} {actual} {expected} {atol}')
    

@pytest.mark.parametrize('likelihood_fn', LIKELIHOOD_FNS_)
def test_num_edits_sampler(likelihood_fn, num_samples=150000, atol=0.002):
    for sequence_length, mutation_rate in PARAMS_TO_TEST_:
        num_edits_sampler = ada_utils.NumberEditsSampler(
            sequence_length, 
            mutation_rate,
            likelihood_fn=likelihood_fn,
            rng_seed=1)
        
        num_edits = num_edits_sampler.sample(num_samples)
        possible_num_edits = np.arange(1, sequence_length + 1)
        
        actual_probs = [float(np.count_nonzero(num_edits == n)) / len(num_edits) 
                        for n in possible_num_edits]
        expected_probs = likelihood_fn(possible_num_edits, sequence_length, mutation_rate)
        
        np.testing.assert_allclose(actual_probs, expected_probs, atol=atol)
    

@pytest.mark.parametrize('likelihood_fn,expected_num_edits_fn', 
                         [(ada_utils.num_edits_likelihood_adabeam, ada_utils.expected_num_edits_adalead_v2),
                          ])
def test_expected_num_edits_adalead(likelihood_fn, expected_num_edits_fn, num_samples=150000, atol=0.002):
    """Tests that the expected number of edits is correct."""
    for sequence_length, mutation_rate in PARAMS_TO_TEST_:
        num_edits_sampler = ada_utils.NumberEditsSampler(
            sequence_length, 
            mutation_rate, 
            likelihood_fn=likelihood_fn,
            rng_seed=1)
        actual = np.mean(num_edits_sampler.sample(num_samples))
        expected = expected_num_edits_fn(sequence_length, mutation_rate)
        
        np.testing.assert_allclose(actual, expected, atol=atol)
            
            
def test_no_tism_cost_fail():
    model_fn = testing_utils.CountLetterModel(
        flip_sign=True,
        vocab_i=0,  # A
    )
    
    model = ada_utils.ModelWrapper(model_fn)
    with pytest.raises(ValueError):
        model.get_tism('ACAAA', idxs=None)


@pytest.mark.parametrize('idx_option', [None, 'all', 'skipC', 'includeC'])
def test_get_tisms_basic(idx_option):
    """Test basic functionality of get_tism."""
    model_fn = testing_utils.CountLetterModel(
        flip_sign=True,
        vocab_i=0,  # A - counts 'A's, so more A's = higher score
    )
    idxs = {
        None: None,
        'all': list(range(5)),
        'skipC': [0, 3, 4],
        'includeC': [0, 1, 2],
    }[idx_option]
    
    model = ada_utils.ModelWrapper(model_fn, tism_cost=1.0)
    sequence = 'ACAAA'
    
    pos_and_chars, logits = model.get_tism(
        sequence=sequence,
        idxs=idxs,
    )
    
    # Check return types
    assert isinstance(pos_and_chars, list)
    assert isinstance(logits, np.ndarray)
    assert logits.dtype == np.float32
    
    # Check structure
    assert len(pos_and_chars) == len(logits)
    for pos, char in pos_and_chars:
        assert isinstance(pos, int)
        assert isinstance(char, str)
        assert char in ['A', 'C', 'G', 'T']
    
    # Check that we don't include mutations to the same character
    if idxs is None:
        positions_to_check = list(range(len(sequence)))
    else:
        positions_to_check = idxs
    
    for pos in positions_to_check:
        base_char = sequence[pos]
        # Should not have (pos, base_char) in results
        assert (pos, base_char) not in pos_and_chars, \
            f"Position {pos} should not have mutation to its own base '{base_char}'"
    
    # Check that we get expected number of mutations
    # For each position, we should have 3 mutations (4 vocab - 1 base)
    expected_num_mutations = len(positions_to_check) * 3
    assert len(pos_and_chars) == expected_num_mutations, \
        f"Expected {expected_num_mutations} mutations, got {len(pos_and_chars)}"


def test_get_tisms_with_idxs():
    """Test get_tisms with specific indices."""
    model_fn = testing_utils.CountLetterModel(
        flip_sign=True,
        vocab_i=0,  # A
    )
    
    model = ada_utils.ModelWrapper(model_fn, tism_cost=1.0)
    sequence = 'ACAAA'
    idxs = [0, 2, 4]  # Only check positions 0, 2, 4
    
    pos_and_chars, logits = model.get_tism(
        sequence=sequence,
        idxs=idxs,
    )
    
    # Check that all positions in results are from idxs
    result_positions = {pos for pos, _ in pos_and_chars}
    assert result_positions.issubset(set(idxs)), \
        f"All result positions should be in {idxs}, but got {result_positions}"
    
    # Check that we have mutations for all specified positions
    for pos in idxs:
        # Should have 3 mutations per position (4 vocab - 1 base)
        mutations_at_pos = [p for p, _ in pos_and_chars if p == pos]
        assert len(mutations_at_pos) == 3, \
            f"Position {pos} should have 3 mutations, got {len(mutations_at_pos)}"