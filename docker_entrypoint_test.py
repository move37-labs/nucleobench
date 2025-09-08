"""Test docker entrypoint.

To test:
```zsh
pytest -n auto docker_entrypoint_test.py --durations=0
```
"""

import argparse
import itertools
import numpy as np
import os
import pytest
import tempfile

from nucleobench import models
from nucleobench import optimizations
from nucleobench.optimizations import model_class as mc
from nucleobench.common import argparse_lib
from nucleobench.common import testing_utils

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
    opt_class = optimizations.get_optimization(optimization)
    _ = opt_class.init_parser()
    init_args = opt_class.debug_init_args()
    init_args['positions_to_mutate'] = [0, 1]
    opt_obj = opt_class(**init_args)

    # Check that obj has required run functions.
    opt_obj.run(n_steps=1)


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
    np.random.seed(0)
    pos_to_mutate = sorted(
        [int(x) for x in np.random.choice(range(200), size=128, replace=False)])
    opt_init_args["positions_to_mutate"] = pos_to_mutate
    if model == "malinois":
        opt_init_args["start_sequence"] = "AT" * 100
    elif model == 'bpnet':
        opt_init_args["start_sequence"] = "AT" * 1000
    elif model == 'enformer':
        opt_init_args["start_sequence"] = "A" * 82_000
    else:
        opt_init_args["start_sequence"] = "AT" * 100
    
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
                    start_sequence=opt_init_args["start_sequence"],
                    positions_to_mutate=opt_init_args["positions_to_mutate"],
                ),
                model_init_args=None,
                opt_init_args=None,
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
    opt_init_args["start_sequence"] = "A" * seq_len
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
                ),
                model_init_args=None,
                opt_init_args=None,
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
            '--start_sequence', f'local://{seed_seq_filename}',
            '--positions_to_mutate', f'local://{pos_to_mutate_filename}',
            ])
        assert parsed_args.main_args.output_path == 'dont use'
        assert len(parsed_args.main_args.start_sequence) == 196_608
        assert parsed_args.main_args.positions_to_mutate == positions_to_mutate


@pytest.mark.parametrize("pos_to_mutate_type", ['empty', 'none', 'local', 'url'])
def test_empty_positions_to_mutate(pos_to_mutate_type):
    """Check that parsing handles positions_to_mutate correctly."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        pos_to_mutate_filename = os.path.join(tmpdirname, 'pos_to_mutate.txt')
        with open(pos_to_mutate_filename, 'w') as f:
            f.write('\n'.join(map(str, range(100))))
        pos_to_mutate = {
            'empty': '',
            'none': None,
            'local': f'local://{pos_to_mutate_filename}',
            'url': 'enformer://12',
        }[pos_to_mutate_type]
        model_fn, opt, parsed_args = de.parse_all([
            '--model', 'dummy',
            '--optimization', 'dummy',
            '--output_path', 'dont use',
            '--start_sequence', 'AAA',
            '--positions_to_mutate', pos_to_mutate,
            ])
        if pos_to_mutate_type in ['empty', 'none']:
            assert parsed_args.main_args.positions_to_mutate == None
        elif pos_to_mutate_type == 'local':
            assert isinstance(parsed_args.main_args.positions_to_mutate, list)
            for p in parsed_args.main_args.positions_to_mutate:
                assert isinstance(p, int)


def test_no_intermediate_records():
    model = "malinois"
    optimization = "directed_evolution"

    model_class = models.get_model(model)
    opt_class = optimizations.get_optimization(optimization)
    
    model_obj = model_class(**model_class.debug_init_args())

    opt_init_args = opt_class.debug_init_args()
    opt_init_args["model_fn"] = model_obj
    if model == "malinois":
        opt_init_args["start_sequence"] = "AT" * 100
    elif model == 'bpnet':
        opt_init_args["start_sequence"] = "AT" * 1000
    elif model == 'enformer':
        opt_init_args["start_sequence"] = "A" * 82_000
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
                    start_sequence=opt_init_args["start_sequence"],
                    positions_to_mutate=None,
                ),
                model_init_args=None,
                opt_init_args=None,
            ),
            ignore_errors=False,
        )

@pytest.mark.parametrize("optimization", _valid_opts)
def test_optimization_pos_to_mutate(optimization):
    """Test that the optimization respects the positions_to_mutate argument."""
    if optimization in optimizations.SAMPLING_IGNORES_POSITIONS_TO_MUTATE_:
        return
    
    seq_len = 1000
    start_sequence = 'A' * seq_len
    np.random.seed(0)
    pos_to_mutate = sorted(
        [int(x) for x in np.random.choice(range(seq_len), size=256, replace=False)])
    
    opt_class = optimizations.get_optimization(optimization)
    _ = opt_class.init_parser()
    init_args = opt_class.debug_init_args()
    init_args['positions_to_mutate'] = pos_to_mutate
    init_args['start_sequence'] = start_sequence
    opt_obj = opt_class(**init_args)

    # Check that obj has required run functions.
    for _ in range(5):
        opt_obj.run(n_steps=2)
    
        # Check that all algorithms obey the `positions_to_mutate` argument.
        proposal = opt_obj.get_samples(1)[0]
        testing_utils.assert_proposal_respects_positions_to_mutate(
            start_sequence, proposal, pos_to_mutate)