"""Tests for gradabeam.py

To test:
```zsh
pytest nucleobench/optimizations/ada/gradabeam/gradabeam_test.py
```
"""

from .gradabeam import GradaBeam


def test_gradabeam_sanity():
    kwargs = GradaBeam.debug_init_args()
    kwargs["debug"] = True

    gradabeam = GradaBeam(**kwargs)

    gradabeam.run(n_steps=2)

    out_seqs = gradabeam.get_samples(kwargs["beam_size"])
    del out_seqs
