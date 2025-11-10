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
    init_args = ledidi.Ledidi.debug_init_args()
    init_args['positions_to_mutate'] = [0] if positions_to_mutate else None
    ledidi.Ledidi(**init_args)


def test_get_samples():
    ld_opt = ledidi.Ledidi(**ledidi.Ledidi.debug_init_args())
    
    for num_samples in [2, 3]:
        ret = ld_opt.get_samples(num_samples)
        assert len(ret) == num_samples
        for st in ret:
            assert st == 'AA'
        

@pytest.mark.parametrize('positions_to_mutate', [False, True])
def test_correctness(positions_to_mutate):
    # Counts 'C'
    init_args = ledidi.Ledidi.debug_init_args()
    init_args['model_fn'] = testing_utils.CountLetterModel(flip_sign=True, vocab_i=1)
    init_args['positions_to_mutate'] = [0] if positions_to_mutate else None
    init_args['lr'] = 1.0
    init_args['train_batch_size'] = 4
    led_opt = ledidi.Ledidi(**init_args)

    start_params = led_opt.designer.weights.detach().clone().numpy()
    start_energy = init_args['model_fn'].inference_on_strings(['AA'])[0]

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
        
        
def test_respects_positions_to_mutate():
    """Test that the optimization respects the positions_to_mutate argument."""
    np.random.seed(0)
    torch.manual_seed(0)
    
    seq_len = 1000
    start_sequence = "".join(np.random.choice(list("ACGT"), size=seq_len))
    
    positions_to_mutate = sorted(
        list(np.random.choice(range(seq_len), size=500, replace=False)))
    
    init_args = ledidi.Ledidi.debug_init_args()
    init_args['start_sequence'] = start_sequence
    init_args['positions_to_mutate'] = positions_to_mutate
    init_args['lr'] = 10.0
    init_args['model_fn'] = testing_utils.CountLetterModel(flip_sign=True)
    led_opt = ledidi.Ledidi(**init_args)
    
    for _ in range(10):
        led_opt.run(n_steps=1)
        for proposal in led_opt.get_samples(5):
            testing_utils.assert_proposal_respects_positions_to_mutate(
                start_sequence, proposal, positions_to_mutate)
            print(f'proposal: {proposal}')