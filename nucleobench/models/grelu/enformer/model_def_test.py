"""Tests for model_def.py

To test:
```zsh
pytest nucleobench/models/grelu/enformer/model_def_test.py
```
"""

import pytest
import random
import torch

from nucleobench.common import testing_utils

from nucleobench.models.grelu.enformer import model_def


model_args = {
    'add_unsqueeze_to_output': True, 
    'call_is_on_strings': False,
    'flip_sign': False,
    'extra_channels': 5313 - 1,
    'add_unsqueeze_to_output': True,
    'train_seq_len': 196_608,
    }


@pytest.mark.parametrize('aggregation_type', ['muscle_CAGE', 'muscle_not_liver'])
def test_model_def_sanity(aggregation_type):
    enformer_args = model_def.Enformer.debug_init_args()
    enformer_args['aggregation_type'] = aggregation_type
    m = model_def.Enformer(
        override_model=testing_utils.CountLetterModel(**model_args),
        **enformer_args)
    ret = m.inference_on_strings(['A' * 196608, 'C' * 196608, 'T' * 196608])
    assert list(ret.shape) == [3]


def _aggregation(model_out: torch.Tensor) -> torch.Tensor:
    assert model_out.ndim == 3
    return model_out[:, 0, 0]


def test_tism_correctness():
    """Check that TISM on an C-count network knows that Cs are important."""
    m = model_def.Enformer(
        override_model=testing_utils.CountLetterModel(**model_args),
        override_aggregation=_aggregation,
        **model_def.Enformer.debug_init_args())
    random.seed(0)
    base_str = [random.choice(['A', 'C', 'T', 'G']) for _ in range(196_608)]
    _, tism = m.tism(base_str)
    for base_nt, tism_dict in zip(base_str, tism):
        assert base_nt not in tism_dict
        if base_nt == 'C':
            # Everything should be the same.
            assert tism_dict['A'] == tism_dict['T'] == tism_dict['G']
            assert tism_dict['A'] > 0  # decrease the count, increase the energy.
        else:
            # TISM should show that the greatest change comes from adding a 'C'.
            for nt in ['A', 'T', 'G']:
                if nt == base_nt: continue
                assert tism_dict[nt] == 0  # changing to a non-C should be no change.
            assert tism_dict['C'] < 0
            

def test_tism_consistency():
    """TISM on a single nucleotide should be the same as the string.."""
    m = model_def.Enformer(
        override_model=testing_utils.CountLetterModel(**model_args),
        **model_def.Enformer.debug_init_args())
    random.seed(0)
    base_str = [random.choice(['A', 'C', 'T', 'G']) for _ in range(196_608)]
    
    v1, tism1 = m.tism(base_str)
    single_bp_tisms = [m.tism(base_str, idx) for idx in range(10)]
    
    for idx in range(len(single_bp_tisms)):
        v2, tism2 = single_bp_tisms[idx]
        assert v1 == v2
        assert len(tism2) == 1
        for k, v in tism2[0].items():
            assert v == tism1[idx][k]