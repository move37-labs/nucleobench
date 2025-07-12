"""
Copyright 2020 Dyno Therapeutics

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
This code was adapted from the following source:
https://github.com/samsinai/FLEXS/blob/master/flexs/baselines/explorers/adalead.py

It has been modified to conform to the nucleobench optimization class interface, and to remove the
dependence on pandas.
"""

from typing import Optional

from nucleobench.common import testing_utils
from nucleobench.common import constants
import argparse
import numpy as np
import random

from nucleobench.optimizations import optimization_class as oc

from nucleobench.optimizations.typing import ModelType, SequenceType, SamplesType
from nucleobench.optimizations.ada import ada_utils


class AdaLeadRef(oc.SequenceOptimizer):
    """
    Adalead explorer.

    Algorithm works as follows:
        Initialize set of top sequences whose fitnesses are at least
            (1 - threshold) of the maximum fitness so far
        While we can still make model queries in this batch
            Recombine top sequences and append to parents
            Rollout from parents and append to mutants.

    """

    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        group = parser.add_argument_group("AdaLead init args")

        group.add_argument(
            "--sequences_batch_size",
            type=int,
            default=10,
            required=False,
            help="Number of sequences to propose for measurement from ground truth per round",
        )
        group.add_argument(
            "--model_queries_per_batch",
            type=int,
            default=100,
            required=False,
            help="Number of allowed in-silico model evaluations per round",
        )
        group.add_argument(
            "--mutation_rate",
            type=float,
            required=False,
            help="Chance of a mutation per residue. "
            "(Original adalead uses mu = mutation_rate * sequence length.)",
        )
        group.add_argument(
            "--recombination_rate",
            type=float,
            required=False,
            help="The probability of a crossover occurring at any position in a sequence",
        )
        group.add_argument(
            "--threshold",
            type=float,
            default=0.2,
            required=False,
            help="In each round only sequences with fitness above (1 - threshold) * f_max "
            "are retained as parents for generating next set of sequences",
        )
        group.add_argument(
            "--rho",
            type=int,
            default=2,
            required=False,
            help="The number of rounds of pairwise recombinations",
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
            "sequences_batch_size": 2,
            "model_queries_per_batch": 10,
            "mutation_rate": 0.9,
            "recombination_rate": 0.0,
            "threshold": 0.25,
            "rho": 0,
            "eval_batch_size": 1,
            "rng_seed": 42,
        }

    def __init__(
        self,
        model_fn: ModelType,
        start_sequence: SequenceType,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        threshold: float,
        rho: int,
        eval_batch_size: int,
        rng_seed: int,
        mutation_rate: float,
        recombination_rate: float,
        positions_to_mutate: Optional[list[int]] = None,
        debug: bool = False,
    ):  
        self.model = ada_utils.ModelWrapper(model_fn)
        self.start_sequence = start_sequence
        self.sequences_batch_size = sequences_batch_size
        self.model_queries_per_batch = model_queries_per_batch
        self.threshold = threshold
        self.recomb_rate = recombination_rate
        self.alphabet = "".join(constants.VOCAB)
        self.mutation_rate = mutation_rate
        self.rho = rho
        self.eval_batch_size = eval_batch_size
        self.rng = random.Random(rng_seed)
        self.positions_to_mutate = positions_to_mutate or list(range(len(start_sequence)))
        self.debug = debug

        assert min(self.positions_to_mutate) >= 0
        assert max(self.positions_to_mutate) < len(start_sequence)

        # For now we expect to receive a single str, that we mutate to create a population.
        assert isinstance(start_sequence, str)
        self.seed_population = [
            ada_utils.generate_random_mutant(
                sequence=start_sequence, 
                positions_to_mutate=self.positions_to_mutate, 
                mu=mutation_rate, 
                alphabet=self.alphabet, 
                rng=self.rng,
            )
            for _ in range(sequences_batch_size)
        ]
        self.seed_scores = np.array(
            [self.model.get_fitness([s])[0] for s in self.seed_population]
        )
        for score in self.seed_scores:
            assert not np.isnan(score)

        self.current_population = [seq for seq in self.seed_population]
        self.current_scores = self.seed_scores.copy()

    @staticmethod
    def run_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        return parser

    @staticmethod
    def debug_run_args():
        return {}

    def run(
        self,
        n_steps: int,
    ):
        for _step in range(n_steps):
            self.current_population, self.current_scores = self.propose_sequences(
                self.current_population, self.current_scores)
        print(f'Current scores: {self.current_scores}')

    def get_samples(self, n_samples: int) -> SamplesType:
        """Get samples."""
        limit = min(n_samples, len(self.current_population))
        return self.current_population[:limit]

    def is_finished(self) -> bool:
        return False

    def propose_sequences(
        self, in_seqs: list[str], in_seq_scores: np.ndarray
    ) -> tuple[list[str], np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""
        measured_sequence_set = set(in_seqs)

        # Get all sequences within `self.threshold` percentile of the top_fitness.
        top_fitness = in_seq_scores.max()
        parent_mask = in_seq_scores >= top_fitness * (
            1 - np.sign(top_fitness) * self.threshold
        )
        parent_inds = np.argwhere(parent_mask).flatten()
        parents = [in_seqs[i] for i in parent_inds]
        if self.debug:
            print(f'After thresholding, went from {len(parent_inds)} to {len(parents)}')

        sequences = {}
        previous_model_cost = self.model.cost
        while self.model.cost - previous_model_cost < self.model_queries_per_batch:
            # Generate recombinant mutants.
            for i in range(self.rho):
                parents = ada_utils.recombine_population(
                    gen=parents, rng=self.rng, recomb_rate=self.recomb_rate, 
                    positions_to_mutate=self.positions_to_mutate)

            for i in range(0, len(parents), self.eval_batch_size):
                # Here we do rollouts from each parent (root of rollout tree).
                roots = parents[i : i + self.eval_batch_size]
                root_fitnesses = self.model.get_fitness(roots)

                nodes = list(enumerate(roots))

                rollout_length = 0
                while (
                    len(nodes) > 0
                    and self.model.cost - previous_model_cost + self.eval_batch_size
                    < self.model_queries_per_batch
                ):
                    child_idxs = []
                    children = []
                    round_num = 0
                    while len(children) < len(nodes):
                        round_num += 1
                        idx, node = nodes[len(children) - 1]

                        child = ada_utils.generate_random_mutant(
                            sequence=node,
                            positions_to_mutate=self.positions_to_mutate,
                            mu=self.mutation_rate,
                            alphabet=self.alphabet,
                            rng=self.rng,
                        )

                        # Stop when we generate new child that has never been seen before.
                        if (
                            child not in measured_sequence_set
                            and child not in sequences
                        ):
                            child_idxs.append(idx)
                            children.append(child)
                        else:
                            pass
                    if self.debug:
                        print(f'It took {round_num} tries to generate a child.')
                    
                    # Stop the rollout once the child has worse predicted
                    # fitness than the root of the rollout tree.
                    # Otherwise, set node = child and add child to the list
                    # of sequences to propose.
                    fitnesses = self.model.get_fitness(children)
                    sequences.update(zip(children, fitnesses))

                    nodes = []
                    for idx, child, fitness in zip(child_idxs, children, fitnesses):
                        if fitness >= root_fitnesses[idx]:
                            nodes.append((idx, child))
                    rollout_length += 1
                
                if self.debug:
                    print(f'Rollout length: {rollout_length}')

        if len(sequences) == 0:
            raise ValueError(
                "No sequences generated. If `model_queries_per_batch` is small, try "
                "making `eval_batch_size` smaller"
            )

        # We propose the top `self.sequences_batch_size` new sequences we have generated.
        if self.debug:
            print(f'Proposing {self.sequences_batch_size} sequences out of {len(sequences)}')
        new_seqs = list(sequences.keys())
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[: -self.sequences_batch_size : -1]
        new_seqs = [new_seqs[i] for i in sorted_order]

        return new_seqs, preds[sorted_order]