"""A simple script to demonstrate simple usage of NucleoBench in Python.

To run:
```zsh
python -m recipes.python.adabeam_substringcount
```
"""

"""Initialize the task."""
from nucleobench import models
# Design for a simple task: count the number of occurances of a particular substring.
# See `nucleobench.models.__init__` for a registry of tasks, or add your own.
model_obj = models.get_model('substring_count')

# Every task has some baseline, default arguments to initialize. We can use
# these to demonstrate, or modify them for custom behavior. We do both, to
# demonstrate.
model_init_args = model_obj.debug_init_args()
model_init_args['substring'] = 'ATGTC'
model_fn = model_obj(**model_init_args)

"""Initialize the designer."""
from nucleobench import optimizations
# Pick a design algorithm that attemps to solve the task. In this case,
# maximize the number of substrings.
opt_obj = optimizations.get_optimization('adabeam')
# Every task has some baseline, default arguments to initialize. We can use
# these to demonstrate, or modify them for custom behavior. We do both, to
# demonstrate.
opt_init_args = opt_obj.debug_init_args()
opt_init_args['model_fn'] = model_fn
opt_init_args['seed_sequence'] = 'A' * 100
designer = opt_obj(**opt_init_args)

"""Run the designer and show the results."""
designer.run(n_steps=100)
ret = designer.get_samples(1)
ret_score = model_fn(ret)
print(f'Final score: {ret_score[0]}')
print(f'Final sequence: {ret[0]}')
