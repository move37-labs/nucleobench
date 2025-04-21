"""A Docker entrpoint that runs multiple, heterogenous model/task types in sequence.

`entrypoint.py` is the preferred entrypoint.

The main use-case for this entrypoint is to run convergence tests correctly. In
the testing framework, we cannot guarantee running on the same machine. To solve it,
we run this entrypoint on a machine and run all algos for a given task on the same
machine, guaranteeing the same compute hardware.

To run:
```zsh
python -m docker_entrypoint_multi \
    --output_path=./docker_entrypoint_test/malinois \
    --common_args=" \
        --model=malinois,\
            --target_feature=0,\
            --bending_factor=1.0,\
        --seed_sequence=GATAAGTGACACGGTGCAGCTCGGGTATCGTCTACGGGTGAAAACGGAAGGGTTCTATCCCATGTGGCCTGCTGACCTACGCACGATAATGAGCATTTAAGTAAGTCGGTGGGCTTTCACATGTTTACCGTCGGGCTCGAAGGCGGGTCCGGAAAACTAATTTCGGATCACCCTACCCAGGACGAACGTCGGGGGTGGCC,\
        --max_seconds=15,\
        --ignore_errors=False\
        " \
    --beam_search_args=" \
        --optimization=beam_search,\
            --beam_size=2,\
            --init_order_method=sequential,\
        --optimization_steps_per_output=1,\
        --proposals_per_round=2,\
        --output_path=beam_search
        " \
    --ledidi_args=" \
        --optimization=ledidi,\
            --train_batch_size=32,\
            --lr=0.1,\
        --optimization_steps_per_output=1,\
        --proposals_per_round=2,\
        --output_path=ledidi
        "
```
"""

import argparse
import os
import sys

import docker_entrypoint


OPT_NAMES_ = [
    'adalead', 'beam_search', 'beam_search_unordered', 'evo', 'evo_tism', 'fastseqprop', 'ledidi', 
    'simulated_annealing']

DICT_TO_STRING_DELIMETER_ = ','

def parse_common_args(argv_list: list):
    parser = argparse.ArgumentParser(description="Entrypoint-multi parser.", add_help=False)
    group = parser.add_argument_group('Main args')
    group.add_argument('--common_args', type=str, required=True, help='')
    group.add_argument('--output_path', type=str, required=True, help='')
    
    for opt_name in OPT_NAMES_:
        group.add_argument(f'--{opt_name}_args', type=str, required=False, help='')

    known_args, leftover_args = parser.parse_known_args(argv_list)
    return known_args, leftover_args


def cmdstr2list(cmd_arg_str: str) -> list[str]:
    """Map string to list of strings exactly like Python cmd."""
    return [x.strip() for x in cmd_arg_str.split(DICT_TO_STRING_DELIMETER_)]


def main(dry_run: bool = False):
    # Extract common args from the list.
    known_args, leftover_args = parse_common_args(sys.argv[1:])
    assert known_args.common_args
    assert known_args.output_path
    assert len(leftover_args) == 0, leftover_args
    common_args = known_args.common_args
    
    # Iteratively "parse" specific args.
    specific_args = []
    for opt_name in OPT_NAMES_:
        arg_str = getattr(known_args, f'{opt_name}_args', None)
        if arg_str not in  ['', None]:
            specific_args.append(arg_str)
    
    # For each of [specific arg1, specific arg2], parse (common args, specific argN)
    # *exactly* as we would in the primary endpoint. 
    run_loop_args_l = []
    for specific_arg in specific_args:
        cmd_arg_list = cmdstr2list(common_args + DICT_TO_STRING_DELIMETER_ + specific_arg)
        print(cmd_arg_list)
        model_fn, opt, all_args = docker_entrypoint.parse_all(cmd_arg_list)
        # Adjust `output_path`.
        all_args.main_args.output_path = os.path.join(known_args.output_path, all_args.main_args.output_path)
        run_loop_args_l.append({
            'model': model_fn,
            'opt': opt,
            'args': all_args,
            'ignore_errors': all_args.main_args.ignore_errors,
        })
    
    if dry_run: return True
    for run_loop_args in run_loop_args_l:
        print('[main] Running...')
        docker_entrypoint.run_loop(**run_loop_args)
    
if __name__ == '__main__':
    main()