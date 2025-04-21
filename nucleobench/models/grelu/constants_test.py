"""Tests for constants.

To test:
```zsh
pytest nucleobench/models/grelu/constants_test.py
```
"""

import numpy as np
import random

import grelu.sequence.format

from nucleobench.models.grelu import constants
from nucleobench.common import string_utils

def test_vocab_consistency():
    """Check that gRelu and NucleoBench use the same vocab."""
    valid_vocab = constants.VOCAB_
    for _ in range(100):
        random_string = "".join(random.choices(valid_vocab, k=100))
        print(random_string)
        grelu_ints = grelu.sequence.format.strings_to_one_hot(random_string)
        nucleobench_ints = string_utils.dna2tensor(random_string, vocab_list=valid_vocab).numpy()
        
        assert np.array_equal(grelu_ints, nucleobench_ints), (random_string, valid_vocab, grelu_ints, nucleobench_ints)