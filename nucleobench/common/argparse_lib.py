"""Utilities for parsing arguments."""

from typing import Iterable

import argparse
import dataclasses

@dataclasses.dataclass
class ParsedArgs:
    main_args: argparse.Namespace
    model_init_args: argparse.Namespace
    opt_init_args: argparse.Namespace
    opt_run_args: argparse.Namespace
    
    
def parse_long_start_sequence(known_args: argparse.Namespace) -> argparse.Namespace:
    """Parse a long start sequence from a file."""
    assert known_args.start_sequence.startswith('local://')
    local_fileloc = known_args.start_sequence[len('local://'):]
    with open(local_fileloc, 'r') as f:
        known_args.start_sequence = f.read()
    return known_args


def possibly_parse_positions_to_mutate(known_args: argparse.Namespace) -> argparse.Namespace:
    """Possibly parse `positions_to_mutate` from a file, or leave it untouched, depending on the value."""
    if isinstance(known_args.positions_to_mutate, str) and known_args.positions_to_mutate.startswith('local://'):
        local_fileloc = known_args.positions_to_mutate[len('local://'):]
        with open(local_fileloc, 'r') as f:
            loc_str = f.read()
        known_args.positions_to_mutate = [int(x) for x in loc_str.split('\n')]
    elif known_args.positions_to_mutate is None or known_args.positions_to_mutate == '' or known_args.positions_to_mutate == []:
        known_args.positions_to_mutate = None
    else:
        assert isinstance(known_args.positions_to_mutate, list), (type(known_args.positions_to_mutate), known_args.positions_to_mutate)
        known_args.positions_to_mutate = [int(x) for x in known_args.positions_to_mutate.split(',')]
    return known_args


def handle_leftover_args(known_args: argparse.Namespace, leftover_args: Iterable):
    """Handle leftover arguments, either by failing or by ignoring them."""
    if known_args.ignore_empty_cmd_args:
        # Check that every "value" is either `None` or `empty`. If so, allow it to continue.
        for i in leftover_args:
            if i.startswith('--'):
                if '=' in i:
                    arg_val = i.split('=')[1]
                    if arg_val not in [None, '']:
                        raise ValueError(f'Unused arg, not empty: {leftover_args}')
                continue
            else: 
                if i not in [None, '']:
                    raise ValueError(f'Unused arg, not empty: {leftover_args}')
    else:
        raise ValueError(f'Unused args: {leftover_args}')
    

def str_to_bool(s):
    if s.lower() in ('yes', 'true', 't', '1'):
        return True
    elif s.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')