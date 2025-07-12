"""Beam search."""

from typing import Generator, Optional, Union

import argparse
import itertools
import numpy as np
import random

from nucleobench.common import argparse_lib
from nucleobench.common import constants
from nucleobench.common import memory_utils
from nucleobench.common import priority_queue
from nucleobench.common import testing_utils

from nucleobench.optimizations import optimization_class as oc
from nucleobench.optimizations.typing import PositionsToMutateType, SequenceType, SamplesType, TISMModelClass

from nucleobench.optimizations.beam_search import beam_utils


INIT_ORDER_METHODS_ = ['sequential', 'random', 'tism_fixed', 'tism_reverse']
class OrderedBeamSearch(oc.SequenceOptimizer):
    """Beam search to minimize model_fn on sequences.."""
        
    def __init__(
        self, 
        model_fn: TISMModelClass, 
        start_sequence: SequenceType,
        beam_size: int,
        init_order_method: str,
        positions_to_mutate: Optional[PositionsToMutateType] = None,
        rng_seed: int = 0,
        # TODO(joelshor): Delete priority queue from here.
        use_priority_queue: bool = False,
        priority_queue_size: int = 32,
        vocab: list[str] = constants.VOCAB,
        minibatch_size: int = 1,
        ):
        assert isinstance(start_sequence, str)
        assert init_order_method in INIT_ORDER_METHODS_
        self.positions_to_mutate = positions_to_mutate or list(range(len(start_sequence)))
        assert min(self.positions_to_mutate) >= 0
        assert max(self.positions_to_mutate) < len(start_sequence)
        
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        
        self.model_fn = model_fn
        self.beam_size = beam_size
        self.start_sequence = start_sequence
        
        self.init_order_method = init_order_method
        
        self.vocab = vocab
        self.minibatch_size = minibatch_size  # minibatch size for network.
        self.use_priority_queue = use_priority_queue
        self.priority_queue_size = priority_queue_size
        
        # Set up the initial queue with the right number of elements.
        # TODO(joelshor): Consider using a priority tree, but probably not,
        # since it's not that helpful.
        self.seed_energy = self.model_fn([self.start_sequence])[0]
        self.beam = beam_utils.Beam(max_items=self.beam_size)
        self.beam.put([(self.seed_energy, self.start_sequence)])
        
        # Optionally set up the queue for the best seen.
        if self.use_priority_queue:
            self.q = priority_queue.OneSidedPriorityQueue(max_items=self.priority_queue_size)
            self.q.push(priority_queue.SearchQItem(
                # Higher is better in the queue, so flip the sign.
                fitness=-1 * self.seed_energy,
                state=self.start_sequence,
                num_edits=0,
            ))
        
        def _get_order(tism, fn):
            order = [(i, fn(t.values())) for i, t in enumerate(tism) if i in self.positions_to_mutate]
            order = sorted(order, key=lambda x: x[1])
            return [x[0] for x in order]
        if self.init_order_method == 'sequential':
            self._search_order = self.positions_to_mutate or list(range(len(self.start_sequence)))
        elif self.init_order_method == 'random':
            self._search_order = self.positions_to_mutate or list(range(len(self.start_sequence)))
            random.shuffle(self._search_order)
        elif self.init_order_method in ['tism_fixed', 'tism_reverse']:
            _, tism = model_fn.tism(self.start_sequence)
            self._search_order = _get_order(tism, min)
            if self.init_order_method == 'tism_reverse':
                self._search_order = self._search_order[::-1]
        else:
            raise ValueError('Arg not recognized.')
        self.n_edits = 0
        
    
    def run(self, n_steps: int):
        if self.n_edits >= len(self.start_sequence):
            return
        last_index_this_run = min(len(self.start_sequence), self.n_edits + n_steps)
        locations_to_edit_this_run = self._search_order[self.n_edits: last_index_this_run]
        
        for loc_to_edit in locations_to_edit_this_run:
            potential_moves = get_potential_moves(self.beam, loc_to_edit, self.vocab)
            evaluated_potential_moves = self._evaluate_potential_moves(list(potential_moves))
            # Lower priority is better.
            evaluated_potential_moves = sorted(evaluated_potential_moves)
            new_beam_items = evaluated_potential_moves[:self.beam_size]
            
            # Add items to the "best seen so far" queue, if applicable.
            if self.use_priority_queue:
                self.q.push_batch([priority_queue.SearchQItem(
                    # Higher is better in the queue.
                    fitness=-1 * e,
                    state=v,
                    num_edits=self.n_edits + 1,
                    ) for e, v in evaluated_potential_moves])
            
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

        # `search_cost_method` is ISM.
        rets = []
        for batch_input in batchify(potential_moves):
            rets.append(self.model_fn(batch_input))
        rets = np.concatenate(rets, axis=0)
        assert rets.shape == (len(potential_moves),), (rets.shape, len(potential_moves))
        rets = zip(rets, potential_moves)
        
        return rets
            
    def _evaluate_potential_moves_generator(
        self, potential_moves: Generator[str, None, None]) -> list[tuple[float, str]]:

        # `search_cost_method` is ISM.
        rets = []
        while True:
            cur_minibatch = list(itertools.islice(potential_moves, self.minibatch_size))
            if cur_minibatch == []:
                break
            rets.append(self.model_fn(cur_minibatch))
        rets = np.concatenate(rets, axis=0)
        rets = zip(rets, potential_moves)
        
        return rets
    
    def get_samples(self, n_samples: int) -> SamplesType:
        """Return subset of elements from the beam."""
        if self.use_priority_queue:
            if n_samples > self.priority_queue_size:
                raise ValueError(f'Number of samples requested too large: {n_samples} {self.priority_queue_size}')
            return self.q.get(n_samples)
        else:
            if n_samples > self.beam_size:
                raise ValueError(f'Number of samples requested too large: {n_samples} {len(self.beam.beam)}')
            return list(self.beam.get_items())[:n_samples]
    
    @property
    def is_batch_mode(self) -> bool:
        return self._is_batch_mode

    @is_batch_mode.setter
    def is_batch_mode(self, is_batch_mode: bool):
        self._is_batch_mode = is_batch_mode

    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        group = parser.add_argument_group('Beam search init args')
        
        group.add_argument('--beam_size', type=int, required=True, help='')
        
        group.add_argument('--init_order_method', type=str, default='tism_fixed', help='',
                           choices=INIT_ORDER_METHODS_)
        group.add_argument('--minibatch_size', type=int, default=1, help='')
        group.add_argument('--use_priority_queue', type=argparse_lib.str_to_bool, default=False, help='')
        group.add_argument('--priority_queue_size', type=int, default=32, help='')
        group.add_argument('--rng_seed', type=int, default=0, required=False, help='')
        
        return parser
    
    @staticmethod
    def debug_init_args():
        return {
            'model_fn': testing_utils.CountLetterModel(),
            'start_sequence': 'AA',
            'beam_size': 5,
            'init_order_method': 'sequential',
            'minibatch_size': 1,
            'use_priority_queue': True,
            'priority_queue_size': 4,
        }

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
    

def get_potential_moves(beam: beam_utils.Beam, loc_to_edit: int, 
                        vocab: list[str]) -> Generator[str, None, None]:
    """Return a generator of potential next strings."""
    for cur_str in beam.get_items():
        for c in vocab:
            yield cur_str[:loc_to_edit] + c + cur_str[loc_to_edit + 1:]