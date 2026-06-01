"""Tests for model_def.py

To test:
```zsh
pytest nucleobench/models/grelu/borzoi/model_def_test.py
```
"""

import pytest
import torch

from nucleobench.common import testing_utils
from nucleobench.models.grelu.borzoi import constants, model_def
from nucleobench.models.grelu.enformer import constants as enformer_constants

model_args = {
    'add_unsqueeze_to_output': True,
    'call_is_on_strings': False,
    'flip_sign': False,
    'extra_channels': 7610,
    'train_seq_len': 524288,
    }


def override_aggregation(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 3
    assert x.shape[1:] == (7611, 1)
    ret = x[:, 0, 0]
    # Return 1D tensor [batch]
    return ret


def test_model_def_sanity():
    m = model_def.Borzoi(
        override_model=testing_utils.CountLetterModel(**model_args),
        override_aggregation=override_aggregation,
        **model_def.Borzoi.debug_init_args())
    ret = m.inference_on_strings(['A' * 524288, 'C' * 524288, 'T' * 524288])
    assert list(ret.shape) == [3]


def test_tism_correctness():
    """Check that TISM on an C-count network knows that Cs are important."""
    m = model_def.Borzoi(
        override_model=testing_utils.CountLetterModel(**model_args),
        override_aggregation=override_aggregation,
        **model_def.Borzoi.debug_init_args())
    base_str = 'A' * 524288
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
                if nt == base_nt:
                    continue
                assert tism_dict[nt] == 0  # changing to a non-C should be no change.
            assert tism_dict['C'] < 0


@pytest.mark.skip
def test_tism_consistency():
    """TISM on a single nucleotide should be the same as the string.."""
    m = model_def.Borzoi(
        override_model=testing_utils.CountLetterModel(**model_args),
        override_aggregation=override_aggregation,
        **model_def.Borzoi.debug_init_args())
    base_str = 'A' * 524288
    v1, tism1 = m.tism(base_str)
    single_bp_tisms = [m.tism(base_str, idx) for idx in range(len(base_str))]

    for idx in range(len(single_bp_tisms)):
        v2, tism2 = single_bp_tisms[idx]
        assert v1 == v2
        assert len(tism2) == 1
        for k, v in tism2[0].items():
            assert v == tism1[idx][k]


def test_inject_middle_sequence():
    """Tests that inject_middle_sequence correctly modifies the sequence."""
    borzoi_len = constants.BORZOI_TRAIN_LEN_
    enformer_len = enformer_constants.ENFORMER_TRAIN_LEN_

    base_sequence = 'A' * borzoi_len
    middle_sequence = 'C' * enformer_len

    modified_sequence = model_def.Borzoi.inject_middle_sequence(
        base_sequence, middle_sequence)

    # Check that the number of 'A's is correct after injection
    expected_a_count = borzoi_len - enformer_len
    actual_a_count = modified_sequence.count('A')
    assert actual_a_count == expected_a_count

    # Verify with CountLetterModel
    model_args = {
        'vocab_i': 0,  # Count 'A's
        'call_is_on_strings': False,
        'train_seq_len': borzoi_len,
        'add_unsqueeze_to_output': True,
        'flip_sign': True,
        'extra_channels': len(constants.BORZOI_TASKS_) - 1,
    }

    # Check that the Borzoi model correctly counts the number of 'A's.
    borzoi_args = model_def.Borzoi.debug_init_args()
    borzoi_args['override_model'] = testing_utils.CountLetterModel(**model_args)
    borzoi_args['override_aggregation'] = override_aggregation
    m = model_def.Borzoi(**borzoi_args)
    result = m([modified_sequence])
    assert result.item() == expected_a_count

    # Check that the counting just the Enformer bins perfectly aligns with the injection.
    borzoi_args['spatial_bins_to_aggregate'] = model_def.Borzoi.enformer_spatial_bins()
    m_middle = model_def.Borzoi(**borzoi_args)
    result_middle = m_middle([modified_sequence])
    assert result_middle.item() == expected_a_count
