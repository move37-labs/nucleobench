"""Test testing_utils.py

To test:
```zsh
pytest nucleobench/common/testing_utils_test.py
```
"""
import random

import pytest

from nucleobench.common import string_utils, testing_utils


def test_dummy_inference_correctness():
    """Test that dummy net properly counts chars."""
    seq_len = 7
    batch_size = 6
    vocab = ['A', 'C', 'G', 'T']
    for vocab_i, letter_to_count in enumerate(vocab):
        seqs = []
        for _ in range(batch_size):
            seqs.append(''.join(random.choices(vocab, k=seq_len)))
        #m = testing_utils.CountLetterModel(vocab_i=vocab_i, **model_args)
        m = testing_utils.CountLetterModel(vocab_i=vocab_i, call_is_on_strings=False)
        tensors = string_utils.dna2tensor_batch(seqs, vocab_list=vocab)
        ret = m(tensors)
        for seq, count in zip(seqs, ret):
             assert seq.count(letter_to_count) == count


def test_tism():
    m = testing_utils.CountLetterModel(vocab_i=1)  # Count C
    _, tism = m.tism('ACTG')

    for base_c, cur_tism in zip('ACTG', tism):
        if base_c == 'C':
            assert cur_tism['A'] == cur_tism['T'] == cur_tism['G'] < 0
        else:
            for nt in ['A', 'T', 'G']:
                if nt == base_c:
                    continue
                assert cur_tism[nt] == 0
            cur_tism['C'] > 0

def test_tism_bp_consistency():
    """Test that single BP tism."""
    m = testing_utils.CountLetterModel(vocab_i=1)  # Count C
    _, tism = m.tism('ACT')

    for idx in range(3):
        _, tism_cur = m.tism('ACT', [idx])
        assert tism[idx] == tism_cur[0]

@pytest.mark.parametrize("vocab_i", [0, 1, 2, 3])
@pytest.mark.parametrize("flip_sign", [True, False])
@pytest.mark.parametrize("extra_channels", [0, 5])
@pytest.mark.parametrize("add_unsqueeze_to_output", [True, False])
@pytest.mark.parametrize("aggregate", [True, False])
def test_count_letter_model_args(vocab_i, flip_sign, extra_channels, add_unsqueeze_to_output, aggregate):
    """Test all possible arguments for CountLetterModel."""
    seq_len = 10
    batch_size = 4
    vocab = ['A', 'C', 'G', 'T']
    letter_to_count = vocab[vocab_i]

    seqs = [''.join(random.choices(vocab, k=seq_len)) for _ in range(batch_size)]

    m = testing_utils.CountLetterModel(
        vocab_i=vocab_i,
        flip_sign=flip_sign,
        extra_channels=extra_channels,
        add_unsqueeze_to_output=add_unsqueeze_to_output,
        aggregate=aggregate,
        call_is_on_strings=False
    )

    tensors = string_utils.dna2tensor_batch(seqs, vocab_list=vocab)
    ret = m(tensors)

    # Check shape
    expected_shape = [batch_size]
    if extra_channels > 0:
        expected_shape.append(extra_channels + 1)
    if not aggregate:
        expected_shape.append(seq_len)
    if add_unsqueeze_to_output:
        expected_shape.append(1)

    assert list(ret.shape) == expected_shape

    # Check values
    for i, seq in enumerate(seqs):
        expected_val = seq.count(letter_to_count)
        if flip_sign:
            expected_val *= -1

        val_to_check = ret[i]
        if extra_channels > 0:
            val_to_check = val_to_check[0] # The count is in the first channel

        if aggregate:
            assert val_to_check.sum() == expected_val
        else:
            assert val_to_check.sum() == expected_val
            # Check individual positions
            for j, char in enumerate(seq):
                expected_pos_val = 1 if char == letter_to_count else 0
                if flip_sign:
                    expected_pos_val *= -1
                assert val_to_check[j] == expected_pos_val
