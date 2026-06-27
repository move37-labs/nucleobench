"""Tests for model_def.py
To test:
```zsh
pytest nucleobench/models/rna/saluki/model_def_test.py
```
"""

import numpy as np
import pytest

from nucleobench.models.rna.saluki import model_def


def test_saluki_sanity():
    """Basic sanity test with override model."""

    # Mock model that returns a list of zeros of length of inputs
    def mock_predict_5utr(seqs):
        return np.zeros((len(seqs), 1))

    m = model_def.SalukiModel(override_model=mock_predict_5utr)
    ret = m.inference_on_strings(["AAA", "CCC", "TTT", "GGG", "ACT"])
    assert list(ret.shape) == [5]
    assert np.all(ret == 0.0)


def test_saluki_call_raises_on_string():
    """Test that __call__ raises ValueError when given a single string."""

    def mock_predict_5utr(seqs):
        return np.zeros((len(seqs), 1))

    m = model_def.SalukiModel(override_model=mock_predict_5utr)
    with pytest.raises(ValueError, match="needs to be list of strings"):
        m("AAA")


def test_debug_init_args():
    """Test debug_init_args static method."""
    args = model_def.SalukiModel.debug_init_args()
    assert isinstance(args, dict)
