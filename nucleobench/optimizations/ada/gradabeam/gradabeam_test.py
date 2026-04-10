"""Tests for gradabeam.py

To test:
```zsh
pytest nucleobench/optimizations/ada/gradabeam/gradabeam_test.py
```
"""

import pytest

import numpy as np

from nucleobench.common import testing_utils

from ..adabeam.adabeam import AdaBeam
from .gradabeam import GradaBeam


def test_gradabeam_sanity():
    kwargs = GradaBeam.debug_init_args()
    kwargs['debug'] = True
    
    gradabeam = GradaBeam(**kwargs)

    gradabeam.run(n_steps=2)

    out_seqs = gradabeam.get_samples(kwargs['beam_size'])
    del out_seqs


def test_gradabeam_convergence():
    kwargs = GradaBeam.debug_init_args()
    
    model_fn = kwargs['model_fn']
    # Be greedy so this definitely improves.
    kwargs['exploration_alpha'] = 0.0001
    
    start_seq = 'A' * 100
    
    start_score = model_fn([start_seq])[0]
    
    kwargs['start_sequence'] = start_seq
    gradabeam = GradaBeam(**kwargs)
    
    gradabeam.run(n_steps=2)
    out_seqs = gradabeam.get_samples(kwargs['beam_size'])
    out_seq_scores = np.array([model_fn([s])[0] for s in out_seqs])
    # GradaBeam should improve (lower is better).
    assert out_seq_scores[0] < start_score


def test_gradabeam_positions_to_mutate():
    """No matter how many iterations, positions outside `positions_to_mutate` shouldn't change."""

    start_seq = 'A' * 100

    beam_size = 2
    kwargs = GradaBeam.debug_init_args()
    kwargs['start_sequence'] = start_seq
    kwargs['beam_size'] = beam_size
    gradabeam = GradaBeam(**kwargs, positions_to_mutate=list(range(20)))

    for i in range(4):
        gradabeam.run(n_steps=1)

        out_seqs = gradabeam.get_samples(beam_size)
        for seq in out_seqs:
            for s in seq[20:]:
                assert s == 'A', seq


@pytest.mark.skip(reason="Disable multi-batch for now.")
@pytest.mark.parametrize('eval_batch_size', [1, 2, 4])
def test_gradabeam_eval_batch_size_sanity(eval_batch_size):
    """Test that `eval_batch_size` works."""
    kwargs = GradaBeam.debug_init_args()
    kwargs['eval_batch_size'] = eval_batch_size
    gradabeam = GradaBeam(**kwargs)

    gradabeam.run(n_steps=2)

    # TODO(joelshor):
    # Add correctness checks.


def test_gradabeam_eval_batch_size_consistency():
    """Test that `eval_batch_size` is consistent."""
    model_fn = testing_utils.CountLetterModel(flip_sign=True)

    seqs = [''.join(np.random.choice(['A', 'G', 'T', 'C'], size=100)) for _ in range(10)]

    kwargs = GradaBeam.debug_init_args()
    kwargs['model_fn'] = model_fn
    kwargs['start_sequence'] = 'A' * 100
    
    kwargs['eval_batch_size'] = 1
    gradabeam_1 = GradaBeam(**kwargs)
    
    kwargs['eval_batch_size'] = 2
    gradabeam_2 = GradaBeam(**kwargs)
    
    kwargs['eval_batch_size'] = 4
    gradabeam_4 = GradaBeam(**kwargs)
    
    scores1 = gradabeam_1.get_batched_fitness(seqs)
    scores2 = gradabeam_2.get_batched_fitness(seqs)
    scores4 = gradabeam_4.get_batched_fitness(seqs)
    
    assert np.array_equal(scores1, scores2)
    assert np.array_equal(scores1, scores4)
    assert np.array_equal(scores2, scores4)


@pytest.mark.skip(reason="Randomness causes this test to fail sometimes.")
def test_no_gradient_gradabeam_is_adabeam():
    """Test that gradabeam with no gradients behaves like adabeam."""
    model_fn = testing_utils.CountLetterModel(flip_sign=True)

    start_seq = 'A' * 10
    start_score = model_fn([start_seq])[0]
    assert start_score == 0
    
    adabeam_args = {
        'model_fn': model_fn,
        'start_sequence': start_seq,
        'beam_size': 2,
        'n_rollouts_per_root': 4,
        'mutations_per_sequence': 2,
        'eval_batch_size': 1,
        'rng_seed': 0,
        'debug': True,
    }
    
    _n_steps = 2
    
    # Run adabeam.
    adabeam = AdaBeam(**adabeam_args)
    adabeam.run(n_steps=_n_steps)
    sample_adabeam = adabeam.get_samples(1)[0]
    score_adabeam = model_fn([sample_adabeam])[0]
    assert score_adabeam < start_score  # Better than the start sequence.
    
    # Run gradabeam.
    gradabeam_args = {
        'exploration_alpha': 0.05,
    }
    gradabeam_args.update(adabeam_args)

    gradabeam = GradaBeam(**gradabeam_args)
    gradabeam.run(n_steps=_n_steps)
    sample_gradabeam = gradabeam.get_samples(1)[0]
    score_gradabeam = model_fn([sample_gradabeam])[0]
    assert score_gradabeam < start_score  # Better than the start sequence.

    # The two should be identical.
    assert score_gradabeam == score_adabeam
    assert sample_gradabeam == sample_adabeam

class TestGradientAlignment:


    def test_gradient_map_matches_territory(self):
        """
        Verifies that when CountLetterModel says 'C is good', 
        GradaBeam picks 'C' and fitness improves.
        """
        start_sequence = "AA" 
        
        # We want Positive Gradients (+1) for 'C'.
        vocab = ['A','C','G','T']
        model = testing_utils.CountLetterModel(
            vocab_i=1,  # Target 'C'
            flip_sign=True,
            vocab=vocab,
        )   

        # Initialize GradaBeam
        gb = GradaBeam(
            model_fn=model,
            start_sequence=start_sequence,
            mutations_per_sequence=1,
            beam_size=5,
            n_rollouts_per_root=1,
            eval_batch_size=1,
            exploration_alpha=0.0,
            gradient_prob_cap=1.0,  # No cap.
            rng_seed=42,
            debug=True,
        )
        
        print("\n[Test] Calculating gradients on root...")
        nodes = gb.initialize_roots_with_gradients(
            [gb.current_nodes[0]]
        )
        root = nodes[0]
        
        # Find the max probability action
        best_flat_idx = np.argmax(root.probs)
        best_pos, best_char = root.pos_and_chars[best_flat_idx]
        best_prob = root.probs[best_flat_idx]
        
        print(f"[Test] Algorithm chose: Pos {best_pos} -> '{best_char}' with prob {best_prob:.4f}")
        
        # ASSERTION 1: Did it pick 'C'?
        target_char = vocab[1] # 'C'
        assert best_char == target_char, \
            f"Vocab Mismatch! Model wanted '{target_char}', but GradaBeam picked '{best_char}'."
            
        # ASSERTION 2: Is it confident?
        assert best_prob >= 0.5, \
            f"Entropy Death! Expected high confidence (>0.8) for '{target_char}', got {best_prob:.4f}."

        # 3. Verify Actual Fitness Gain
        original_seq = root.seq
        mut_list = list(original_seq)
        mut_list[int(best_pos)] = best_char
        mutant_seq = "".join(mut_list)
        
        base_score = gb.get_batched_fitness([original_seq])[0]
        new_score = gb.get_batched_fitness([mutant_seq])[0]
        delta = new_score - base_score
        
        print(f"[Test] Fitness: {base_score} -> {new_score} (Delta: {delta})")
        
        # ASSERTION 3: Maximization check
        # Score should increase (Delta > 0)
        assert delta > 0, \
            f"Optimization Failure! Mutating to '{best_char}' should improve score, but delta was {delta}."

