"""Tests for ribosome_loading.py

To test:
```zsh
pytest nucleobench/models/rna/rinalmo_mrl/rinalmo/ribosome_loading_test.py
```
"""

import torch

from .data.alphabet import Alphabet
from .ribosome_loading import RibosomeLoadingPredictionWrapper


def test_rinalmo_sanity():
    """Basic sanity test."""
    model = RibosomeLoadingPredictionWrapper(force_cpu=True)
    model.to('cpu')
    model.eval()

    seqs = ['AC' * 8, 'TG' * 8]
    encoded_seq = Alphabet().batch_tokenize(seqs)
    tokens = torch.tensor(encoded_seq, dtype=torch.int64, device='cpu')
    with torch.no_grad():
        outputs = model(tokens)
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (2,)
