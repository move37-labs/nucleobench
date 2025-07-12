"""AdaBeam.

Adaptive beam, adaptive mutation rate, adaptive directed evolution.
"""

from typing import Optional

import argparse
import numpy as np

from nucleobench.common import argparse_lib
from nucleobench.common import constants
from nucleobench.common import testing_utils

from nucleobench.optimizations import optimization_class as oc
from nucleobench.optimizations.typing import ModelType, SequenceType, SamplesType

from nucleobench.optimizations.ada import ada_utils


RolloutNode = ada_utils.RolloutNode


class AdaBeam(oc.SequenceOptimizer):
    """AdaBeam designer."""

    def __init__(
        self,
        model_fn: ModelType,
        start_sequence: SequenceType,
        mutations_per_sequence: float,
        beam_size: int,
        n_rollouts_per_root: int,
        eval_batch_size: int,
        skip_repeat_sequences: bool = False,
        rng_seed: int = 0,
        positions_to_mutate: Optional[list[int]] = None,
        max_rollout_len: int = 200,
        debug: bool = False,
    ):
        """AdaBeam nucleic acid sequence designer.

        Args:
            model_fn: The model function to use for scoring sequences. Sometimes called the "oracle" or "task.
            start_sequence: Start sequence for the optimization.
            mutations_per_sequence: The expected number of mutations per sequence. Actual number of mutations
                per round is sampled from a distribution.
            beam_size: Maximum number of candidates to carry from one round to the next.
            n_rollouts_per_root: Number of explorations per root, per round. 
            eval_batch_size: Number of sequences to run inference on the model at once.
            rng_seed: Seed for the pseudo-random number generator.
            skip_repeat_sequences: If `True`, skip sequences that have already been evaluated. If `False`,
                don't skip, but use the cached value if it exists.
            positions_to_mutate: Optional list of positions to mutate in the seed sequence.
                If `None`, all positions are considered.
            max_rollout_len: Maximum number of rollouts to perform per parent node.
            debug: If `True`, print debug information.
        """
        self.positions_to_mutate = positions_to_mutate or list(
            range(len(start_sequence))
        )
        assert min(self.positions_to_mutate) >= 0
        assert max(self.positions_to_mutate) < len(start_sequence)

        assert mutations_per_sequence > 0  # 0 NOT allowed.
        assert mutations_per_sequence <= len(self.positions_to_mutate)
        assert beam_size > 0

        # If we do zero rollouts per parent, we will have no child nodes.
        assert n_rollouts_per_root > 0
        
        self.model = ada_utils.ModelWrapper(model_fn, use_cache=True, debug=debug)

        self.skip_repeat_sequences = skip_repeat_sequences
        self.start_sequence = start_sequence
        self.beam_size = beam_size
        self.n_rollouts_per_root = n_rollouts_per_root
        self.alphabet = "".join(constants.VOCAB)
        self.mu = float(mutations_per_sequence) / len(self.positions_to_mutate)
        self.eval_batch_size = eval_batch_size
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)
        self.num_mutations_sampler = self.get_sampler(self.mu)
        self.max_rollout_len = max_rollout_len
        
        self.debug = debug

        # Mutate a string to create a starting population.
        assert isinstance(start_sequence, str)
        seed_node = RolloutNode(seq=start_sequence, fitness=None)
        num_edit_locs = self.num_mutations_sampler.sample(beam_size)
        self.current_nodes = []
        for i in range(0, beam_size, self.eval_batch_size):
            cur_num_edits = num_edit_locs[i : i + self.eval_batch_size]
            self.current_nodes.extend(
                self.mutate_nodes(
                    [seed_node] * len(cur_num_edits),
                    cur_num_edits,
            ))


    def get_batched_fitness(self, sequences: list[str]) -> np.ndarray:
        """Get fitness for a batch of sequences."""
        return ada_utils.get_batched_fitness(
            model_wrapper=self.model,
            sequences=sequences,
            batch_size=self.eval_batch_size,
        )


    def generate_mutations(self, sequence: str, random_n_locs: int) -> str:
        """Convenience wrapper."""
        return ada_utils.generate_random_mutant_v2(
            sequence=sequence,
            positions_to_mutate=self.positions_to_mutate,
            random_n_loc=random_n_locs,
            alphabet=self.alphabet,
            rng=self.rng,
        )


    def get_sampler(self, mu: float) -> ada_utils.NumberEditsSampler:
        """Get a sampler for the number of mutations."""
        return ada_utils.NumberEditsSampler(
            sequence_len=len(self.positions_to_mutate), 
            mutation_rate=mu,
            likelihood_fn=ada_utils.num_edits_likelihood_adalead_v2,
            rng_seed=self.rng_seed,
        )
        
    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        group = parser.add_argument_group("AdaLead init args")

        group.add_argument(
            "--beam_size",
            type=int,
            default=10,
            required=False,
            help="Number of sequences to propose for measurement from ground truth per round",
        )
        group.add_argument(
            "--mutations_per_sequence",
            type=float,
            required=False,
            help="The expected number of mutations per sequence.",
        )
        group.add_argument(
            "--n_rollouts_per_root",
            type=int,
            default=4,
            required=False,
            help="Number of rollouts to perform per parent node (per round)",
        )
        group.add_argument(
            "--skip_repeat_sequences",
            type=argparse_lib.str_to_bool,
            default=True,
            required=False,
            help="",
        )
        group.add_argument(
            "--eval_batch_size",
            type=int,
            default=1,
            required=False,
            help="For code optimization; size of batches sent to model",
        )
        group.add_argument(
            "--rng_seed",
            type=int,
            default=42,
            required=False,
            help="Seed for the pseudo-random number generator",
        )

        return parser

    @staticmethod
    def debug_init_args():
        return {
            "model_fn": testing_utils.CountLetterModel(),
            "start_sequence": "AAAAAA",
            "beam_size": 10,
            "mutations_per_sequence": 1,
            "n_rollouts_per_root": 4,
            "eval_batch_size": 1,
            "skip_repeat_sequences": False,  # avoids infinite loops.
            "rng_seed": 42,
        }

    @staticmethod
    def run_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        return parser

    @staticmethod
    def debug_run_args():
        return {}

    def run(self, n_steps: int):
        for _step in range(n_steps):
            self.current_nodes = self.propose_sequences(self.current_nodes)
        print(f'Step {_step} current scores: {sorted([x.fitness for x in self.current_nodes], reverse=True)}')

    def get_samples(self, n_samples: int) -> SamplesType:
        """Get samples."""
        limit = min(n_samples, len(self.current_nodes))
        sorted_nodes = sorted(self.current_nodes, key=lambda x: x.fitness, reverse=True)
        return [x.seq for x in sorted_nodes][:limit]

    def is_finished(self) -> bool:
        return False

    def propose_sequences(self, root_nodes: list[RolloutNode]) -> list[RolloutNode]:
        """Propose top `beam_size` sequences for evaluation."""
        sequences, rollout_lengths = set(), []
        # Perform `n_rollouts_per_root` rollouts on each root node.
        root_nodes_effective = root_nodes * self.n_rollouts_per_root
        for i in range(0, len(root_nodes_effective), self.eval_batch_size):
            # Start a rollout from each root.
            cur_root_nodes = root_nodes_effective[i : i + self.eval_batch_size]
            parent_nodes = cur_root_nodes

            # While there are still active rollouts...
            cur_rollout_length = 0
            while len(parent_nodes) > 0 and cur_rollout_length < self.max_rollout_len:
                # Generate the desired number of edits for each child.
                num_edit_locs = self.num_mutations_sampler.sample(len(parent_nodes))
                
                # Generate a mutated child for each node.
                children = self.mutate_nodes(parent_nodes, num_edit_locs)

                # Add these children to the candidate set of new sequences.
                sequences.update(children)

                # Stop the rollout once the child has worse fitness.
                cur_rollout_length += 1
                new_nodes = []
                for child, comparison_node in zip(children, parent_nodes):
                    if child.fitness >= comparison_node.fitness:
                        new_nodes.append(child)
                    else:
                        rollout_lengths.append(cur_rollout_length)
                parent_nodes = new_nodes
         
        if self.debug:           
            print(f'Rollout lengths: {rollout_lengths}')

        if len(sequences) == 0:
            raise ValueError("No sequences generated.")
        
        # Propose the top `self.beam_size` new sequences we have generated.
        if self.debug:
            print(f'Number of candidate sequences: {len(sequences)}')
        sequences = sorted(sequences, key=lambda x: x.fitness, reverse=True)
        top_nodes = sequences[: self.beam_size]
        
        return top_nodes
    
    
    def mutate_nodes(self, 
                     nodes: list[RolloutNode], 
                     num_edit_locs: list[int],
                     max_num_tries: int = 300,
                     ) -> list[RolloutNode]:
        assert len(nodes) == len(num_edit_locs) <= self.eval_batch_size, (len(nodes), len(num_edit_locs), self.eval_batch_size)
        seqs = []
        for n, random_n_loc in zip(nodes, num_edit_locs):
            # If `self.skip_repeat_sequences=True` keep trying until we get a new sequence.
            try_cnt = 0
            while True:
                candidate = self.generate_mutations(n.seq, random_n_loc)
                try_cnt += 1
                if not self.skip_repeat_sequences or not self.model.str_in_cache(candidate):
                    break
                if try_cnt > max_num_tries:
                    raise ValueError(f"Couldn't find unique child after {try_cnt} tries.")
                if self.debug and try_cnt % 50 == 0:
                    print(f'Couldnt find unique child after {try_cnt} tries...')
            if self.debug:
                if try_cnt > 1:
                    print(f"Found child after {try_cnt} tries")
            seqs.append(candidate)
        fitnesses = self.get_batched_fitness(seqs)
        assert len(fitnesses) == len(seqs) == len(nodes) == len(num_edit_locs)
        for f in fitnesses:
            assert not np.isnan(f)
        
        return [RolloutNode(seq=seq, fitness=f) for seq, f in zip(seqs, fitnesses)]
