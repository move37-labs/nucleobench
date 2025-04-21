"""Randomly change sequence."""

import argparse
from typing import Optional

from typing import Callable
from nucleobench.common import constants
from nucleobench.optimizations import optimization_class as oc
import random

class RandomPerturbation(oc.SequenceOptimizer):
    """A dummy optimizer."""

    def __init__(self, 
                 model_fn: Callable, 
                 seed_sequence: str,
                 positions_to_mutate: Optional[list[int]] = None,
                 ):
        del model_fn
        self.seq = list(seed_sequence)
        self.positions_to_mutate = positions_to_mutate
        
    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        return parser
    
    @staticmethod
    def debug_init_args():
        return {
            'model_fn': None,
            'seed_sequence': 'AA',
        }
        
    @staticmethod
    def run_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        return parser

    def run(self, n_steps: int):
        positions = self.positions_to_mutate or list(range(len(self.seq)))
        for _ in range(n_steps):
            i = random.choice(positions)
            new_nt = random.choice(constants.VOCAB)
            self.seq[i] = new_nt
            
    @staticmethod
    def debug_run_args():
        return {}
        
    def get_samples(self, n_samples: int) -> list[str]:
        """Get samples."""
        return [''.join(self.seq)] * n_samples
    
    def is_finished(self) -> bool:
        return False
