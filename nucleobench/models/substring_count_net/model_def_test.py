"""Tests for model_def.py

To test:
```zsh
pytest nucleobench/models/substring_count_net/model_def_test.py
```
"""

import torch

from nucleobench.models.substring_count_net import model_def


def test_inference_correctness():
    m = model_def.CountSubstringModel(
        substring="ATCG",
    )
    
    rets = m(['AAAAAAAAAAAA', 'ATCGATCGATCG', 'ATCAATCAATCA'])
    assert len(rets) == 3
    assert rets[1] > rets[2] > rets[0]
    
    m = model_def.CountSubstringModel(
        substring="ATG",
    )
    assert m(["ATG"])[0] == 3**2
    assert m(["ATGC"])[0] == 3**2
    assert m(["ATGCG"])[0] == 10
    

def test_tism():
    """TISM should find the "mistake" letter in the repeat substring."""
    m = model_def.CountSubstringModel(
        substring="ATC",
        tism_times=2,
        tism_stdev=0.1,
    )
    
    seq = 'ATCATGATC'
    v, tism = m.tism(seq)
    assert v[0] == m([seq])[0]
    
    other_tisms = []
    for i in range(len(seq)):
        if i == 5:
            assert tism[i]['C'] > 0
            assert tism[i]['C'] > tism[i]['A']
            assert tism[i]['C'] > tism[i]['T']
            should_be_max_tism = tism[i]['C']
        else:
            other_tisms.extend(tism[i].values())
            for v in tism[i].values():
                assert v < 0, (i, tism[i])
    assert should_be_max_tism > max(other_tisms)
    
    
def test_tism_sanity():
    """TISM should find the "mistake" letter in the repeat substring."""
    # TODO(joelshor): Make this test better.
    m = model_def.CountSubstringModel(
        substring="ATC",
        tism_times=2,
        tism_stdev=0.1,
    )
    
    seq = 'ATCATGATC'
    for idx in range(len(seq) - 1):
        v, tism = m.tism(seq, [idx, idx+1])


def test_tism_torch_correctness():
    vocab = ['A', 'C']
    # Use a simple substring and sequence
    model = model_def.CountSubstringModel(
        substring='A', 
        tism_times=1, 
        tism_stdev=0.0, 
        vocab=vocab,
        )

    seq = 'AAC'
    # tism_torch should return a tensor of shape (vocab_size, seq_len)
    tism_tensor = model.tism_torch(seq)
    assert isinstance(tism_tensor, torch.Tensor)
    assert tism_tensor.shape == (2, 3)
    # For this model, the smoothgrad is just the gradient of the count wrt input (since tism_times=1, stdev=0)
    # The reference base at each position should be zero
    base_seq_idx = [0, 0, 1]  # 'A', 'A', 'C' in vocab ['A', 'C']
    for i, ref in enumerate(base_seq_idx):
        assert tism_tensor[ref, i] == 0.0
    # The other base should be the difference in smoothgrad between that base and the reference
    # (for this toy model, values may be 0, but we check type and shape)


def test_tism_torch_consistency_with_tism():
    vocab = ['A', 'C']
    
    # Use a simple substring and sequence
    model = model_def.CountSubstringModel(
        substring='A', 
        tism_times=1, 
        tism_stdev=0.0, 
        vocab=vocab,
        )
    seq = 'AAC'
    # tism returns (y, sg_dicts), tism_torch returns tensor
    _, sg_dicts = model.tism(seq)
    tism_tensor = model.tism_torch(seq)
    # Compare values for each base and position
    for i, sg_dict in enumerate(sg_dicts):
        for j, nt in enumerate(vocab):
            # tism_torch: [j, i], sg_dict: nt
            val_torch = float(tism_tensor[j, i])
            val_py = float(sg_dict.get(nt, 0.0))
            assert abs(val_torch - val_py) < 1e-6