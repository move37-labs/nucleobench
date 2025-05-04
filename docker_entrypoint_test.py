"""Test docker entrypoint.

To test:
```zsh
pytest docker_entrypoint_test.py --durations=0
```
"""

import argparse
import itertools
import os
import pytest
import tempfile

from nucleobench import models
from nucleobench import optimizations
from nucleobench.optimizations import model_class as mc
from nucleobench.common import argparse_lib

import docker_entrypoint as de

_valid_models = list(models.MODELS_.keys())
_valid_opts = optimizations.OPTIMIZATIONS_.keys()
_valid_model_opt_pairs = list(itertools.product(_valid_models, _valid_opts))


@pytest.mark.parametrize("model", _valid_models)
def test_model_required_fns(model):
    """Check that models have required run functions."""
    if model == 'enformer':
        seqs = ["A" * 82_000, "T" * 82_000]
    else:
        seqs = ["A" * 200, "T" * 200]
    
    model_class = models.get_model(model)
    _ = model_class.init_parser()
    model_obj = model_class(**model_class.debug_init_args())
    
    model_obj(seqs)


@pytest.mark.parametrize("optimization", _valid_opts)
def test_optimization_required_fns(optimization):
    # TODO(joelshor, erikstran): Figure out why adalead hangs, and unblock it.
    if optimization == 'adalead':
        return
    opt_class = optimizations.get_optimization(optimization)
    _ = opt_class.init_parser()
    init_args = opt_class.debug_init_args()
    init_args['positions_to_mutate'] = [0, 1]
    opt_obj = opt_class(**init_args)

    # Check that obj has required run functions.
    opt_obj.run(n_steps=1, **opt_class.debug_run_args())


@pytest.mark.parametrize("model,optimization", _valid_model_opt_pairs)
def test_run_loop_with_all_combos(model, optimization):
    if optimization in optimizations.OPTIMIZATIONS_REQUIRING_PYTORCH_DIFF_ and not isinstance(
        model, mc.PyTorchDifferentiableModel
    ):
        return
    if optimization in optimizations.OPTIMIZATIONS_REQUIRING_TISM_ and not isinstance(model, mc.TISMModelClass):
        return
    if model == 'enformer':
        # Takes too long to run as a test.
        return

    model_class = models.get_model(model)
    opt_class = optimizations.get_optimization(optimization)
    
    model_obj = model_class(**model_class.debug_init_args())

    opt_init_args = opt_class.debug_init_args()
    opt_init_args["model_fn"] = model_obj
    if model == "malinois":
        opt_init_args["seed_sequence"] = "AT" * 100
    elif model == 'bpnet':
        opt_init_args["seed_sequence"] = "AT" * 1000
    elif model == 'enformer':
        opt_init_args["seed_sequence"] = "A" * 82_000
    opt_obj = opt_class(**opt_init_args)

    with tempfile.TemporaryDirectory() as tmpdirname:
        de.run_loop(
            model=model_obj,
            opt=opt_obj,
            all_args=argparse_lib.ParsedArgs(
                main_args=argparse.Namespace(
                    model=model,
                    optimization=optimization,
                    max_number_of_rounds=1,
                    optimization_steps_per_output=1,
                    proposals_per_round=1,
                    output_path=tmpdirname,
                    trace_memory=False,
                ),
                model_init_args=None,
                opt_init_args=None,
                opt_run_args=argparse.Namespace(**opt_class.debug_run_args()),
            ),
            ignore_errors=False,
        )


@pytest.mark.skip(reason='Not needed, too long.')
@pytest.mark.parametrize("optimization, flank_length",
                         itertools.product(_valid_opts, [0]))
def test_run_loop_with_flank_length(optimization, flank_length):
    model = "malinois"
    model_class = models.get_model(model)
    opt_class = optimizations.get_optimization(optimization)

    model_init_args = model_class.debug_init_args()
    model_init_args['flank_length'] = flank_length
    seq_len = 600 - 2 * flank_length
    model_obj = model_class(**model_init_args)

    opt_init_args = opt_class.debug_init_args()
    opt_init_args["model_fn"] = model_obj
    opt_init_args["seed_sequence"] = "A" * seq_len
    opt_obj = opt_class(**opt_init_args)

    with tempfile.TemporaryDirectory() as tmpdirname:
        de.run_loop(
            model=model_obj,
            opt=opt_obj,
            all_args=argparse_lib.ParsedArgs(
                main_args=argparse.Namespace(
                    model=model,
                    optimization=optimization,
                    max_number_of_rounds=1,
                    optimization_steps_per_output=1,
                    proposals_per_round=1,
                    output_path=tmpdirname,
                    trace_memory=False,
                ),
                model_init_args=None,
                opt_init_args=None,
                opt_run_args=argparse.Namespace(**opt_class.debug_run_args()),
            ),
            ignore_errors=False,
        )

def test_read_seed_sequence_from_local_file():
    """Check that parsing reads local files when necessary."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        seed_seq_filename = os.path.join(tmpdirname, 'long_seed_seq.txt')
        with open(seed_seq_filename, 'w') as f:
            f.write('A' * 196_608)
        pos_to_mutate_filename = os.path.join(tmpdirname, 'pos_to_mutate.txt')
        positions_to_mutate = list(range(0, 10_000, 100))
        with open(pos_to_mutate_filename, 'w') as f:
            f.write('\n'.join([str(x) for x in positions_to_mutate]))
        model_fn, opt, parsed_args = de.parse_all([
            '--model', 'dummy',
            '--optimization', 'dummy',
            '--output_path', 'dont use',
            '--seed_sequence', f'local://{seed_seq_filename}',
            '--positions_to_mutate', f'local://{pos_to_mutate_filename}',
            ])
        assert parsed_args.main_args.output_path == 'dont use'
        assert len(parsed_args.main_args.seed_sequence) == 196_608
        assert parsed_args.main_args.positions_to_mutate == positions_to_mutate


def test_empty_positions_to_mutate():
    model_fn, opt, parsed_args = de.parse_all([
        '--model', 'dummy',
        '--optimization', 'dummy',
        '--output_path', 'dont use',
        '--seed_sequence', 'AAA',
        '--positions_to_mutate', '',
        ])
    assert parsed_args.main_args.positions_to_mutate == None


def test_no_intermediate_records():
    model = "malinois"
    optimization = "directed_evolution"

    model_class = models.get_model(model)
    opt_class = optimizations.get_optimization(optimization)
    
    model_obj = model_class(**model_class.debug_init_args())

    opt_init_args = opt_class.debug_init_args()
    opt_init_args["model_fn"] = model_obj
    if model == "malinois":
        opt_init_args["seed_sequence"] = "AT" * 100
    elif model == 'bpnet':
        opt_init_args["seed_sequence"] = "AT" * 1000
    elif model == 'enformer':
        opt_init_args["seed_sequence"] = "A" * 82_000
    opt_obj = opt_class(**opt_init_args)

    with tempfile.TemporaryDirectory() as tmpdirname:
        de.run_loop(
            model=model_obj,
            opt=opt_obj,
            all_args=argparse_lib.ParsedArgs(
                main_args=argparse.Namespace(
                    model=model,
                    optimization=optimization,
                    max_number_of_rounds=1,
                    optimization_steps_per_output=-1,
                    proposals_per_round=1,
                    output_path=tmpdirname,
                    trace_memory=False,
                ),
                model_init_args=None,
                opt_init_args=None,
                opt_run_args=argparse.Namespace(**opt_class.debug_run_args()),
            ),
            ignore_errors=False,
        )