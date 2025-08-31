"""Entrypoint for docker container.

To test this locally, writing locally:
```zsh
python -m docker_entrypoint \
    --start_sequence GATAAGTGACACGGTGCAGCTCGGGTATCGTCTACGGGTGAAAACGGAAGGGTTCTATCCCATGTGGCCTGCTGACCTACGCACGATAATGAGCATTTAAGTAAGTCGGTGGGCTTTCACATGTTTACCGTCGGGCTCGAAGGCGGGTCCGGAAAACTAATTTCGGATCACCCTACCCAGGACGAACGTCGGGGGTGGCC \
    --model malinois \
        --target_feature 1 \
        --bending_factor 1.0 \
    --optimization fastseqprop \
        --learning_rate 0.1 \
        --eta_min 1e-6 \
        --batch_size 32 \
        --rnd_seed 0 \
    --max_seconds 300  \
    --proposals_per_round 16 \
    --optimization_steps_per_output 1 \
    --output_path ./docker_entrypoint_test/malinois_fs
```
"""
import argparse
import datetime
import os
import sys
import time
import tqdm
from typing import Any

from nucleobench.common import gcp_utils
from nucleobench.common import testing_utils

from nucleobench import models
from nucleobench.common import argparse_lib
from nucleobench import optimizations
from nucleobench.optimizations import model_class as mc
from nucleobench.optimizations import optimization_class as oc


def run_loop(
    model: mc.ModelClass,
    opt: oc.SequenceOptimizer,
    all_args: argparse_lib.ParsedArgs,
    ignore_errors: bool = False,
    ):
    args = all_args.main_args
    
    # Determine whether the end condition should be met by time or number of rounds.
    if args.max_number_of_rounds is None:
        max_time = args.max_seconds
        args.max_number_of_rounds = 99999999
    else:
        if args.max_number_of_rounds == 0:
            args.max_number_of_rounds = 99999999
        max_time = None
        
    # Determine whether to record intermediate steps or not.
    if args.optimization_steps_per_output == -1:
        optimization_steps_per_output_effective = 1
        record_intermediate_steps = False
    else:
        optimization_steps_per_output_effective = args.optimization_steps_per_output
        record_intermediate_steps = True

    # Initialize performance counters.
    opt_time = 0
    all_dicts_to_write = []
    exp_start_time = time.time()
    exp_starttime_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Log a "starting" round for graphs to look reasonable.
    to_write = _get_dict_to_write(
        opt=opt, 
        all_args=all_args, 
        model=model, 
        exp_start_time=exp_start_time, 
        exp_starttime_str=exp_starttime_str, 
        opt_time=opt_time, 
        round_i=-1)
    all_dicts_to_write.append(to_write)

    print('Starting loop...')
    # At the start, write a 'START.txt' file.
    gcp_utils.write_txt_file(args.output_path, content='START')
    
    for round_i in tqdm.tqdm(range(args.max_number_of_rounds)):
        try:
            if opt.is_finished():
                break

            # Update total time.
            cur_total_time = time.time() - exp_start_time
            if max_time is not None and cur_total_time > max_time:
                break

            # Take some optimization steps.
            s_time = time.time()
            opt.run(n_steps=optimization_steps_per_output_effective)
            e_time = time.time()
            opt_time += (e_time - s_time)

            if record_intermediate_steps:
                to_write = _get_dict_to_write(
                    opt=opt, 
                    all_args=all_args, 
                    model=model, 
                    exp_start_time=exp_start_time, 
                    exp_starttime_str=exp_starttime_str, 
                    opt_time=opt_time, 
                    round_i=round_i)
                all_dicts_to_write.append(to_write)

            tot_steps = (round_i+1) * optimization_steps_per_output_effective
            tot_time = time.time() - exp_start_time
            print(f'Completed round {round_i} ({optimization_steps_per_output_effective} steps) took {(e_time - s_time):.2f}s. Avg {tot_time/tot_steps:.2f}s per step.')
        except Exception as e:
            print(e)
            if ignore_errors:
                continue
            else:
                raise e

    # Add one more entry at the end, that includes the proposals.
    to_write = _get_dict_to_write(
        opt=opt, 
        model=model, 
        all_args=all_args, 
        exp_start_time=exp_start_time, 
        exp_starttime_str=exp_starttime_str, 
        opt_time=opt_time, 
        round_i=args.max_number_of_rounds, 
        write_proposals=True)
    all_dicts_to_write.append(to_write)

    # After safe completion, write everything.
    gcp_utils.save_proposals(all_dicts_to_write, args, args.output_path)

    # At the end of it all, write a 'SUCCESS.txt' file.
    gcp_utils.write_txt_file(args.output_path, content='SUCCESS')


def _get_dict_to_write(
    opt,
    all_args: argparse_lib.ParsedArgs,
    model,
    exp_start_time,
    exp_starttime_str,
    opt_time,
    round_i,
    write_proposals: bool = False,
    ) -> dict[str, Any]:
    # Get some proposals.
    proposals = opt.get_samples(all_args.main_args.proposals_per_round)
    
    # Check that positions to mutate are respected.
    if all_args.main_args.optimization not in optimizations.SAMPLING_IGNORES_POSITIONS_TO_MUTATE_:
        try:
            start_sequence = all_args.main_args.start_sequence
            pos_to_mutate = all_args.main_args.positions_to_mutate
        except AttributeError as e:
            raise ValueError(all_args.main_args) from e
        for proposal in proposals:
            testing_utils.assert_proposal_respects_positions_to_mutate(
                start_sequence, proposal, pos_to_mutate)

    # Calculate their energies.
    # TODO(joelshor): Figure out how to add arbitrary debug info. Until then, disable
    # the debug info.
    #energies, debug_info = model(proposals, return_debug_info=True)
    energies = model(proposals)

    # Write output.
    cur_total_time = time.time() - exp_start_time
    to_write = {
        'energies': energies,
        'opt_time': opt_time,
        'exp_starttime_str': exp_starttime_str,
        'total_time': cur_total_time,
        'timestamp': time.strftime("%Y%m%d_%H%M%S"),
        'all_args': all_args,
        'round': round_i,
        'steps': (round_i + 1) * all_args.main_args.optimization_steps_per_output,
    }
    # We no longer write the proposals by default, since they aren't used and blow up output disk size.
    if write_proposals:
        to_write['proposals'] = proposals
    else:
        to_write['proposals'] = [None]

    # TODO(joelshor): Figure out how to add arbitrary debug info. Until then, disable
    # the debug info.
    # to_write.update(debug_info)

    return to_write


def parse_all(argv: list) -> tuple[mc.ModelClass, oc.SequenceOptimizer, argparse_lib.ParsedArgs]:
    parser = argparse.ArgumentParser(description="", add_help=False)
    group = parser.add_argument_group('Main args')

    group.add_argument('--start_sequence', type=str, required=False, help='')
    group.add_argument('--positions_to_mutate', default=None, required=False, help='String for file, or comma delimited list of ints.')
    group.add_argument('--model', type=str, required=True, help='',
                       choices=models.MODELS_.keys())
    group.add_argument('--optimization', type=str, required=True, help='',
                       choices=optimizations.OPTIMIZATIONS_.keys())
    group.add_argument('--output_path', type=str, required=True,
                       help='Directory for all the structured output.')
    group.add_argument(
        '--optimization_steps_per_output', type=int, default=10, 
        help='The number of optimization steps to run before recording a step. Note that `-1` means "do not record any intermediate steps."')
    group.add_argument('--proposals_per_round', type=int, default=1, help='')
    group.add_argument('--ignore_errors', type=argparse_lib.str_to_bool, default=False, help='')
    group.add_argument(
        '--ignore_empty_cmd_args', type=argparse_lib.str_to_bool, default=False, 
        help='Ignore empty commandline args. Useful to have one script for all models/optimizers.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--max_number_of_rounds', type=int, help="Max steps.")
    group.add_argument("--max_seconds", type=int, help="Max time.")

    known_args, leftover_args = parser.parse_known_args(argv)
    model_obj = models.get_model(known_args.model)
    opt_obj = optimizations.get_optimization(known_args.optimization)

    # Now parse additional args, as required by model or optimization method.
    model_init_args, leftover_args = model_obj.init_parser().parse_known_args(leftover_args)
    opt_init_args, leftover_args = opt_obj.init_parser().parse_known_args(leftover_args)
    if len(leftover_args) > 0:
        argparse_lib.handle_leftover_args(known_args, leftover_args)

    # Some start sequences are too long to pass via cmd (they are ~200K bp), so we use 
    # local or gcp files instead.
    if known_args.start_sequence.startswith('local://'):
        known_args = argparse_lib.parse_long_start_sequence(known_args)
    known_args = argparse_lib.possibly_parse_positions_to_mutate(known_args)

    # Initialize objects.
    model_fn = model_obj(**vars(model_init_args))
    opt = opt_obj(
        model_fn=model_fn, 
        start_sequence=known_args.start_sequence, 
        positions_to_mutate=known_args.positions_to_mutate,
        **vars(opt_init_args))

    return model_fn, opt, argparse_lib.ParsedArgs(
        main_args=known_args,
        model_init_args=model_init_args,
        opt_init_args=opt_init_args)


def main(dry_run: bool = False):
    model_fn, opt, all_args = parse_all(sys.argv[1:])
    
    print(f'[main] main_args: {all_args.main_args}')
    print('[main] Running...')

    if dry_run: return True
    run_loop(
        model=model_fn,
        opt=opt,
        all_args=all_args,
        ignore_errors=all_args.main_args.ignore_errors)

if __name__ == "__main__":
    main()