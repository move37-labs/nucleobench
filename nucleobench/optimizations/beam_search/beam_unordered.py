"""Unorded beam search."""

from typing import Optional

import argparse
import numpy as np

from nucleobench.common import constants
from nucleobench.common import memory_utils
from nucleobench.common import testing_utils

from nucleobench.optimizations.typing import PositionsToMutateType, SequenceType, SamplesType, TISMModelClass
from nucleobench.optimizations import optimization_class as oc
from nucleobench.optimizations import utils as opt_utils

from nucleobench.optimizations.beam_search import beam_utils


EDIT_LOCATION_ALGOS_ = ['all', 'random']
EDIT_PROPOSAL_ALGOS_ = ['all', 'random']

class UnorderedBeamSearch(oc.SequenceOptimizer):
    """Beam search to minimize model_fn on sequences.."""
    
    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        group = parser.add_argument_group('Beam search init args')
        
        group.add_argument('--beam_size', type=int, required=True, help='')
        group.add_argument('--edit_location_algo', type=str, default='all', help='',
                           choices=EDIT_LOCATION_ALGOS_)
        group.add_argument('--edit_proposal_algo', type=str, default='all', help='',
                           choices=EDIT_PROPOSAL_ALGOS_)
        group.add_argument('--random_n_loc', type=int, default=-1, help='')
        group.add_argument('--minibatch_size', type=int, default=1, help='')
        group.add_argument('--rng_seed', type=int, default=1, help='')
        
        return parser
    
    @staticmethod
    def debug_init_args():
        return {
            'model_fn': testing_utils.CountLetterModel(),
            'start_sequence': 'AAAA',
            'beam_size': 5,
            'edit_location_algo': 'random',
            'edit_proposal_algo': 'random',
            'random_n_loc': 2,
            'minibatch_size': 1,
        }
    
    def __init__(
        self, 
        model_fn: TISMModelClass, 
        start_sequence: SequenceType,
        beam_size: int,
        edit_location_algo: str,
        edit_proposal_algo: str,
        positions_to_mutate: Optional[PositionsToMutateType] = None,
        random_n_loc: int = -1,
        vocab: list[str] = constants.VOCAB,
        minibatch_size: int = 1,
        rng_seed: int = 0,
        ):
        assert isinstance(start_sequence, str)
        self.positions_to_mutate = positions_to_mutate or list(range(len(start_sequence)))
        assert min(self.positions_to_mutate) >= 0
        assert max(self.positions_to_mutate) < len(start_sequence)
        
        if edit_location_algo not in EDIT_LOCATION_ALGOS_:
            raise ValueError('Arg not recognized.')
        self.edit_location_algo = edit_location_algo
        
        if random_n_loc == -1:
            random_n_loc = len(self.positions_to_mutate)
        assert random_n_loc > 0
        assert random_n_loc <= len(start_sequence)
        self.random_n_loc = random_n_loc
        
        if edit_proposal_algo not in EDIT_PROPOSAL_ALGOS_:
            raise ValueError('Arg not recognized.')
        self.edit_proposal_algo = edit_proposal_algo
        
        
        self.rng = np.random.default_rng(rng_seed)
        
        self.model_fn = model_fn
        self.beam_size = beam_size
        self.start_sequence = start_sequence
        
        self.vocab = vocab
        self.minibatch_size = minibatch_size  # minibatch size for network.
        
        # Set up the initial queue with the right number of elements.
        # TODO(joelshor): Consider using a priority tree, but probably not,
        # since it's not that helpful.
        self.seed_energy = self.model_fn([self.start_sequence])[0]
        self.beam = beam_utils.Beam(max_items=self.beam_size)
        self.beam.put([(self.seed_energy, self.start_sequence)])
        
        self.n_edits = 0

    def run(self, n_steps: int):
        for step_i in range(n_steps):
            
            potential_moves = []
            for base_str in self.beam.get_items():
                # Get potential locations to edit.
                locations_to_edit = opt_utils.get_locations_to_edit(
                    positions_to_mutate=self.positions_to_mutate, 
                    random_n_loc=self.random_n_loc, 
                    rng=self.rng,
                    method=self.edit_location_algo)
                
                # From locations, get deduped potential moves.
                single_edit_mutants = opt_utils.generate_single_edit_mutants(
                    base_str=base_str, 
                    loc_to_edit=locations_to_edit,
                    alphabet=self.vocab,
                    rng=self.rng,
                    method=self.edit_proposal_algo,
                )
                potential_moves.extend(list(set(single_edit_mutants)))

            evaluated_potential_moves = self._evaluate_potential_moves(potential_moves)
            evaluated_potential_moves = list(evaluated_potential_moves)
            
            # Keep the best move in the beam so far.
            evaluated_potential_moves.append(self.beam.get_best_val_and_state())
            
            # Lower priority is better.
            evaluated_potential_moves = sorted(evaluated_potential_moves)
            new_beam_items = evaluated_potential_moves[:self.beam_size]

            # Explicitly clear current beam.
            # TODO(joelshor): Check that this does something, otherwise remove.
            del self.beam.beam
            del self.beam
            del potential_moves
            del evaluated_potential_moves
            memory_utils.free_memory(debug=False)
            
            # Update the new beam.
            self.beam = beam_utils.Beam(max_items=self.beam_size)
            self.beam.put(new_beam_items)
            self.n_edits += 1
            
            
    def _evaluate_potential_moves(self, potential_moves: list[str]) -> list[tuple[float, str]]:
        def batchify(lst):
            """Reshapes a list into batches of a given size."""
            return [lst[i:i + self.minibatch_size] 
                    for i in range(0, len(lst), self.minibatch_size)]

        rets = []
        for batch_input in batchify(potential_moves):
            rets.append(self.model_fn(batch_input))
        rets = np.concatenate(rets, axis=0)
        assert rets.shape == (len(potential_moves),), (rets.shape, len(potential_moves))
        rets = zip(rets, potential_moves)
        
        return rets
    
    def get_samples(self, n_samples: int) -> SamplesType:
        """Return subset of elements from the beam."""
        if n_samples > self.beam_size:
            raise ValueError(f'Number of samples requested too large: {n_samples} {len(self.beam)}')
        return list(self.beam.get_items())[:n_samples]

    @staticmethod
    def run_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        return parser
    
    @staticmethod
    def debug_run_args():
        return {
        }
    
    def is_finished(self) -> bool:
        return self.n_edits >= len(self.start_sequence)