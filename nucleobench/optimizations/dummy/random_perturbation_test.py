"""
Test random perturbation.

To test:
```zsh
pytest nucleobench/optimizations/dummy/random_perturbation_test.py
```
"""
from nucleobench.optimizations.dummy.random_perturbation import RandomPerturbation

def test_sanity():
    dummy_opt = RandomPerturbation(None, 'AAAA', positions_to_mutate=[0])
    for _ in range(4):
        dummy_opt.run(2)
        sample = dummy_opt.get_samples(1)[0]
        assert sample[1:] == 'AAA'