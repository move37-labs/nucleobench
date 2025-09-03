"""Utilities for parsing arguments.

To test that GCP resources exist and can be read:
```zsh
python -m nucleobench.common.argparse_lib
```
"""

from typing import Iterable, Optional, Union

import argparse
import dataclasses
import pandas as pd

GCP_ENFORMER_URL_ = 'https://storage.googleapis.com/nucleobench-start-sequences/start_sequences_enformer.parquet'

@dataclasses.dataclass
class ParsedArgs:
    main_args: argparse.Namespace
    model_init_args: argparse.Namespace
    opt_init_args: argparse.Namespace
    

def possibly_parse_start_sequence(start_seq: str) -> str:
    """Possibly parse start sequence from a local or remote file.
    
    Prefix strings that trigger special handling:
    - `local://`: Load from a local file.
    - `gcp_enformer://[0-9]+`: Load from fixed location for Enformer start sequences, as used
        in the publication.
    """
    if start_seq.startswith('local://'):
        local_fileloc = start_seq[len('local://'):]
        with open(local_fileloc, 'r') as f:
            start_seq = f.read()
    elif start_seq.startswith('gcp_enformer://'):
        # The structure of a GCP enformer string is:
        # `gcp_enformer://[0-9]+`
        # Example:
        # gcp_enformer://12
        index = int(start_seq[len('gcp_enformer://'):])
        df = fetch_gcp_enformer_start_sequence_df()
        start_seq = df[df.index == index]['sequence'].values
        assert len(start_seq) == 1, f"Expected 1 sequence, got {len(start_seq)}"
        start_seq = start_seq[0]
    return start_seq


def fetch_gcp_enformer_start_sequence_df() -> pd.DataFrame:
    """Fetch a start sequence from Zenodo."""
    # Download the CSV file from public GCP bucket.
    return pd.read_parquet(GCP_ENFORMER_URL_)


def possibly_parse_positions_to_mutate(positions_to_mutate: Optional[Union[str, list[int]]]) -> Optional[list[int]]:
    """Possibly parse `positions_to_mutate` from a file, or leave it untouched, depending on the value.
    
    Prefix strings that trigger special handling:
    - `local://`: Load from a local file.
    - `gcp_enformer://[0-9]+`: Load from fixed location for Enformer start sequences, as used
    """
    if isinstance(positions_to_mutate, str) and positions_to_mutate.startswith('local://'):
        local_fileloc = positions_to_mutate[len('local://'):]
        with open(local_fileloc, 'r') as f:
            loc_str = f.read()
        positions_to_mutate = [int(x) for x in loc_str.split('\n')]
    elif isinstance(positions_to_mutate, str) and positions_to_mutate.startswith('gcp_enformer://'):
        # The structure of a GCP enformer string is:
        # `gcp_enformer://[0-9]+`
        # Example:
        # gcp_enformer://12
        index = int(positions_to_mutate[len('gcp_enformer://'):])
        df = fetch_gcp_enformer_start_sequence_df()
        positions_to_mutate = df[df.index == index]['positions_to_mutate'].values
        assert len(positions_to_mutate) == 1, f"Expected 1 positions_to_mutate, got {len(positions_to_mutate)}"
        positions_to_mutate = positions_to_mutate[0].tolist()
        assert isinstance(positions_to_mutate, list), (type(positions_to_mutate), positions_to_mutate)
        positions_to_mutate = [int(x) for x in positions_to_mutate]
    elif positions_to_mutate is None or positions_to_mutate == '' or positions_to_mutate == []:
        positions_to_mutate = None
    else:
        assert isinstance(positions_to_mutate, str)
        positions_to_mutate = [int(x) for x in positions_to_mutate.split(',')]
    return positions_to_mutate


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
    
    
if __name__ == "__main__":
    random_seq = possibly_parse_start_sequence('gcp_enformer://12')
    assert isinstance(random_seq, str), (type(random_seq), random_seq)
    assert len(random_seq) == 196608, f"Expected 196608bp sequence, got {len(random_seq)}"
    print(random_seq)
    
    random_pos_to_mutate = possibly_parse_positions_to_mutate('gcp_enformer://12')
    assert isinstance(random_pos_to_mutate, list)
    assert len(random_pos_to_mutate) == 256
    print(random_pos_to_mutate)