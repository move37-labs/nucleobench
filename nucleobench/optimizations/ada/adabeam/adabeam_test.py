"""Tests for adabeam.py

To test:
```zsh
pytest nucleobench/optimizations/ada/adabeam/adabeam_test.py
```
"""

import pytest

from nucleobench.common import testing_utils

from .adabeam import AdaBeam


@pytest.mark.parametrize("skip_repeat_sequences", [True, False])
def test_adabeam_sanity(skip_repeat_sequences):
    model_fn = testing_utils.CountLetterModel(flip_sign=True)

    start_seq = "A" * 100
    start_score = model_fn([start_seq])[0]
    assert start_score == 0

    beam_size = 20
    kwargs = AdaBeam.debug_init_args()
    kwargs["model_fn"] = model_fn
    kwargs["start_sequence"] = start_seq
    kwargs["beam_size"] = beam_size
    kwargs["skip_repeat_sequences"] = skip_repeat_sequences
    adabeam = AdaBeam(**kwargs)

    adabeam.run(n_steps=2)

    out_seqs = adabeam.get_samples(beam_size)
    del out_seqs
