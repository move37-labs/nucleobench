"""Tests for model_def.py

To test:
```zsh
pytest nucleobench/models/substring_count_net/model_def_test.py
```
"""

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