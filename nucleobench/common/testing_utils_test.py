"""Test testing_utils.py

To test:
```zsh
pytest nucleobench/common/testing_utils_test.py
```
"""
import random

from nucleobench.common import string_utils
from nucleobench.common import testing_utils


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
                if nt == base_c: continue
                assert cur_tism[nt] == 0
            cur_tism['C'] > 0
            
def test_tism_bp_consistency():
    """Test that single BP tism."""
    m = testing_utils.CountLetterModel(vocab_i=1)  # Count C
    _, tism = m.tism('ACT')
    
    for idx in range(3):
        _, tism_cur = m.tism('ACT', [idx])
        assert tism[idx] == tism_cur[0]
    