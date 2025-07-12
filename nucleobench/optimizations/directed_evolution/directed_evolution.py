"""Custom implementation of Fast SeqProp."""

from typing import Callable, Optional

import argparse
import numpy as np
import torch

from nucleobench.common import argparse_lib
from nucleobench.common import constants
from nucleobench.common import testing_utils

from nucleobench.optimizations.typing import PositionsToMutateType, SequenceType, SamplesType, ModelType
from nucleobench.optimizations import optimization_class as oc

from nucleobench.optimizations.directed_evolution import directed_evolution_module as de_mod


class DirectedGreedyEvolution(oc.SequenceOptimizer):
    """Directed greed evolution of a pack of sequences. Based on work from Genentech's gRelu.
    """

    def __init__(self, 
                 model_fn: ModelType, 
                 start_sequence: SequenceType,
                 positions_to_mutate: Optional[PositionsToMutateType] = None,
                 batch_size: int = 1,
                 use_tism: bool = False,
                 location_only: bool = False,
                 budget: int = None,
                 fraction_tism: float = 0.5,
                 rnd_seed: int = 0,
                 vocab: list[str] = constants.VOCAB,
                 verbose: bool = False,
                 ):
        torch.manual_seed(rnd_seed)
        np.random.seed(rnd_seed)
        
        self.rnd_seed = rnd_seed
        self.vocab = vocab
        self.model_fn = model_fn
        self.start_sequence = start_sequence
        self.current_sequence = [start_sequence]
        self.batch_size = batch_size
        self.verbose = verbose
        self.positions_to_mutate = positions_to_mutate
        
        # TISM args.
        if use_tism:
            self.tism_args = de_mod.TISMArgs(
                location_only=location_only,
                budget=budget,
                fraction_tism=fraction_tism,
            )
        else:
            self.tism_args = None
        print(f'Parsed TISM args: {self.tism_args}')
                 
        
    @staticmethod
    def run_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        return parser
    
    @staticmethod
    def debug_run_args():
        return {
        }
        
    def run(self, 
            n_steps: int,
            ) -> list[np.ndarray]:
        """Runs the optimization."""
        
        best_seqs, best_score, energies = de_mod.evolve(
            model=self.model_fn,
            seqs=self.current_sequence,
            max_iter=n_steps,
            batch_size=self.batch_size,
            positions=self.positions_to_mutate,
            verbose=self.verbose,
            tism_args=self.tism_args,
        )
        print(f'Best score: {best_score}')
        del best_score
        
        self.current_sequence = best_seqs
        return energies
        
    
    def get_samples(self, n_samples: int) -> SamplesType:
        """Get samples."""
        del n_samples
        return self.current_sequence
    
    def is_finished(self) -> bool:
        return False
    
    
    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        group = parser.add_argument_group('Directed evolution init args')
        
        group.add_argument('--batch_size', type=int, default=1, required=False, help='')
        group.add_argument('--rnd_seed', type=int, default=0, required=False, help='')
        
        # TISM args.
        group.add_argument('--use_tism', type=argparse_lib.str_to_bool, default=False, required=False, help='')
        group.add_argument('--location_only', type=argparse_lib.str_to_bool, default=False, required=False, help='')
        group.add_argument('--budget', type=int, default=None, required=False, help='')
        group.add_argument('--fraction_tism', type=float, default=0.5, required=False, help='')
        
        return parser
    
    @staticmethod
    def debug_init_args():
        return {
            'model_fn': testing_utils.CountLetterModel(),
            'start_sequence': 'AA',
            'rnd_seed': 0,
        }