"""Tests for adalead utils.

To test:
```zsh
pytest nucleobench/optimizations/ada/ada_utils_test.py
```
"""

import numpy as np
import pytest
import random

import torch

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
    ada_utils.num_edits_likelihood_adalead_v2,
    ada_utils.num_edits_likelihood_shifted_binomial]


@pytest.mark.parametrize('likelihood_fn', LIKELIHOOD_FNS_)
def test_num_edits_likelihood_legacy_prob_dist(likelihood_fn):
    for sequence_length, mutation_rate in PARAMS_TO_TEST_:
        actual_sum = np.sum([likelihood_fn(x, sequence_length, mutation_rate) 
                            for x in range(sequence_length+1)])
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
            num_edits=num_edits,
            seq_len=sequence_length,
            mu=mutation_rate
        ))
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
        possible_num_edits = list(range(1, sequence_length + 1))
        
        actual_probs = [float(np.count_nonzero(num_edits == n)) / len(num_edits) 
                        for n in possible_num_edits]
        expected_probs = [likelihood_fn(n, sequence_length, mutation_rate) for n in possible_num_edits]
        
        np.testing.assert_allclose(actual_probs, expected_probs, atol=atol)
    

@pytest.mark.parametrize('likelihood_fn,expected_num_edits_fn', 
                         [(ada_utils.num_edits_likelihood_adalead_v2, ada_utils.expected_num_edits_adalead_v2),
                          (ada_utils.num_edits_likelihood_shifted_binomial, ada_utils.expected_num_edits_adalead_shifted_binomial),
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
        
   
def test_expected_num_edits_converter(num_samples: int = 100_000, atol: float = 0.005):
    """Test that the calculated p gives the desired expected number of edits."""
    for sequence_length in [10, 100]:
        for expected_num_edits in np.arange(1.5, sequence_length + 1, 2):
            mu = ada_utils.expected_edits_to_p_shifted_binomial(
                expected_number_of_edits=expected_num_edits,
                sequence_len=sequence_length)
            sampler = ada_utils.NumberEditsSampler(
                sequence_len=sequence_length,
                mutation_rate=mu,
                likelihood_fn=ada_utils.num_edits_likelihood_shifted_binomial,
                rng_seed=1
            )
            actual_samples = sampler.sample(num_samples)
            actual_expected = np.mean(actual_samples)
            np.testing.assert_allclose(actual_expected, expected_num_edits, atol=atol)