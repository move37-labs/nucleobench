"""A dummy model to test with."""

import argparse
from typing import Iterable

from nucleobench.common import constants

from nucleobench.optimizations import model_class as mc

class DummyModel(mc.ModelClass):
    def __init__(self, vocab: list[str] = constants.VOCAB, to_count: str = 'A'):
        self.vocab = set(vocab)
        self.to_count = to_count

    @staticmethod
    def init_parser():
        """
        Add model-specific arguments to the argument parser.

        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.

        Returns:
            argparse.ArgumentParser: Argument parser with added model-specific arguments.
        """
        return argparse.ArgumentParser()
    
    @staticmethod
    def debug_init_args():
        return {}
    

    def inference_on_strings(self, seqs: Iterable[str]) -> float:
        """Batch call on list of sequences."""
        if not isinstance(seqs, Iterable):
            raise ValueError(f'Expected `Iterable[str]`, got {type(seqs)}')
        for seq in seqs:
            if not set(seq).issubset(self.vocab):
                raise ValueError(f'Invalid sequence: {seq}')
        
        return [s.count(self.to_count) for s in seqs]
    
    
    def __call__(self, seqs: list[str], return_debug_info: bool = False) -> float:
        ret = self.inference_on_strings(seqs)
        if return_debug_info:
            return ret, {}
        else:
            return ret
        