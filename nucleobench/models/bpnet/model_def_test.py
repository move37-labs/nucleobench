"""Tests for model_def.py

To test:
```zsh
pytest nucleobench/models/bpnet/model_def_test.py
```
"""

from nucleobench.common import testing_utils

from nucleobench.models.bpnet import model_def


model_args = {
    'add_unsqueeze_to_output': True, 
    'call_is_on_strings': False,
    'flip_sign': False}

def test_model_def_sanity():
    m = model_def.BPNet(
        protein='None', 
        override_model=testing_utils.CountLetterModel(**model_args))
    ret = m.inference_on_strings(['AAA', 'CCC', 'TTT', 'GGG', 'ACT'])
    assert list(ret.shape) == [5]
    
    
def test_tism_correctness():
    """Check that TISM on an C-count network knows that Cs are important."""
    m = model_def.BPNet(
        protein='None', 
        override_model=testing_utils.CountLetterModel(**model_args))
    base_str = 'ATCCA'
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
    m = model_def.BPNet(
        protein='None', 
        override_model=testing_utils.CountLetterModel(**model_args))
    base_str = 'ATCCA'
    v1, tism1 = m.tism(base_str)
    single_bp_tisms = [m.tism(base_str, idx) for idx in range(len(base_str))]
    
    for idx in range(len(single_bp_tisms)):
        v2, tism2 = single_bp_tisms[idx]
        assert v1 == v2
        assert len(tism2) == 1
        for k, v in tism2[0].items():
            assert v == tism1[idx][k]