"""
Adaptive beam, adaptive mutation rate, adaptive directed evolotion.
"""

from typing import Optional

from nucleobench.common import testing_utils
from nucleobench.common import constants
import argparse
import numpy as np

from nucleobench.optimizations import optimization_class as oc

from nucleobench.optimizations.ada import ada_utils


SequenceType = str
SamplesType = list[str]


RolloutNode = ada_utils.RolloutNode


class AdaBeam(oc.SequenceOptimizer):
    """AdaBeam designer."""

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
            type=int,
            required=False,
            help="The expected number of mutations per sequence.",
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
            "--n_rollouts_per_root",
            type=int,
            default=4,
            required=False,
            help="Number of rollouts to perform per parent node (per round)",
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
            "seed_sequence": "AAAAAA",
            "beam_size": 10,
            "mutations_per_sequence": 1,
            "threshold": 0.25,
            "n_rollouts_per_root": 4,
            "eval_batch_size": 1,
            "rng_seed": 42,
        }

    def __init__(
        self,
        model_fn: callable,
        seed_sequence: SequenceType,
        mutations_per_sequence: int,
        beam_size: int,
        threshold: float,
        n_rollouts_per_root: int,
        eval_batch_size: int,
        rng_seed: int,
        positions_to_mutate: Optional[list[int]] = None,
        max_rollout_len: int = 200,
        debug: bool = False,
    ):
        """Improved AdaLead (AdaLead v2).

        Args:
            model_fn (callable): _description_
            seed_sequence (SequenceType): _description_
            mutations_per_sequence (int): _description_
            beam_size (int): _description_
            threshold (float): _description_
            n_rollouts_per_root (int): _description_
            eval_batch_size (int): _description_
            rng_seed (int): _description_
            positions_to_mutate (Optional[list[int]], optional): _description_. Defaults to None.
        """
        self.positions_to_mutate = positions_to_mutate or list(
            range(len(seed_sequence))
        )
        assert min(self.positions_to_mutate) >= 0
        assert max(self.positions_to_mutate) < len(seed_sequence)

        assert mutations_per_sequence > 0  # 0 NOT allowed.
        assert mutations_per_sequence <= len(self.positions_to_mutate)

        # If we do zero rollouts per parent, we will have no child nodes.
        assert n_rollouts_per_root > 0

        self.model = ada_utils.ModelWrapper(model_fn)
        self.seed_sequence = seed_sequence
        self.beam_size = beam_size
        self.threshold = threshold
        self.n_rollouts_per_root = n_rollouts_per_root
        self.alphabet = "".join(constants.VOCAB)
        self.mu = float(mutations_per_sequence) / len(self.positions_to_mutate)
        self.eval_batch_size = eval_batch_size
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)
        self.num_mutations_sampler = self.get_sampler(self.mu)
        self.max_rollout_len = max_rollout_len
        
        self.debug = debug

        # For now we expect to receive a single str, that we mutate to create a population.
        assert isinstance(seed_sequence, str)
        num_edit_locs = self.num_mutations_sampler.sample(beam_size)
        self.seed_population = [
            ada_utils.generate_random_mutant_v2(
                sequence=seed_sequence,
                positions_to_mutate=self.positions_to_mutate,
                random_n_loc=random_n_locs,
                alphabet=self.alphabet,
                rng=self.rng,
            )
            for random_n_locs in num_edit_locs
        ]
        # TODO(erikstrand): Use `eval_batch_size` here.
        self.seed_scores = np.array(
            [self.model.get_fitness([s])[0] for s in self.seed_population]
        )
        for score in self.seed_scores:
            assert not np.isnan(score)

        self.current_population = [seq for seq in self.seed_population]
        self.current_scores = self.seed_scores.copy()

    def get_sampler(self, mu: float) -> ada_utils.NumberEditsSampler:
        """Get a sampler for the number of mutations."""
        return ada_utils.NumberEditsSampler(
            sequence_len=len(self.positions_to_mutate), 
            mutation_rate=mu,
            likelihood_fn=ada_utils.num_edits_likelihood_adalead_v2,
            rng_seed=self.rng_seed,
        )

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
                self.current_population, self.current_scores
            )
        print(f"Current scores, mu: {self.current_scores}")

    def get_samples(self, n_samples: int) -> SamplesType:
        """Get samples."""
        limit = min(n_samples, len(self.current_population))
        return self.current_population[:limit]

    def is_finished(self) -> bool:
        return False

    def propose_sequences(
        self, roots: list[str], root_scores: np.ndarray
    ) -> tuple[list[str], np.ndarray]:
        """Propose top `beam_size` sequences for evaluation."""
        roots, root_scores = ada_utils.threshold_on_fitness_percentile(
            roots, root_scores, self.threshold)

        sequences, rollout_lens = [], []
        for _ in range(self.n_rollouts_per_root):
            for i in range(0, len(roots), self.eval_batch_size):
                # Start a rollout from each root.
                parent_seqs = roots[i : i + self.eval_batch_size]
                parent_fitnesses = root_scores[i : i + self.eval_batch_size]
                
                parents = [RolloutNode(seq=seq,
                                       num_edits_from_root=0,
                                       num_edits_from_parent=0,
                                       fitness=fitness,
                                      )
                    for seq, fitness in zip(parent_seqs, parent_fitnesses)
                ]

                # While there are still active rollouts...
                cur_rollout_len = 0
                while len(parents) > 0 and cur_rollout_len < self.max_rollout_len:
                    # Generate a mutated child for each node.
                    children = self.mutate_nodes(parents)

                    # Add these children to the candidate set of new sequences.
                    sequences.extend(children)

                    # Stop the rollout once the child has worse predicted fitness than the parent of
                    # the rollout tree.
                    new_nodes = []
                    for child, parent in zip(children, parents):
                        if child.fitness >= parent.fitness:
                            new_nodes.append(child)

                    parents = new_nodes
                    cur_rollout_len += 1
                rollout_lens.append(cur_rollout_len)
         
        if self.debug:           
            print(f'Rollout lengths: {rollout_lens}')

        if len(sequences) == 0:
            raise ValueError("No sequences generated.")
        
        # Propose the top `self.beam_size` new sequences we have generated.
        sequences = sorted(sequences, key=lambda x: x.fitness, reverse=True)
        top_nodes = sequences[: self.beam_size]
        
        top_seqs = [x.seq for x in top_nodes]
        top_scores = [x.fitness for x in top_nodes]

        return top_seqs, np.array(top_scores)
    
    
    def mutate_nodes(self, nodes: list[RolloutNode]) -> list[RolloutNode]:
        num_edit_locs = self.num_mutations_sampler.sample(len(nodes))
        seqs = [ada_utils.generate_random_mutant_v2(    
                sequence=n.seq,
                positions_to_mutate=self.positions_to_mutate,
                random_n_loc=random_n_loc,
                alphabet=self.alphabet,
                rng=self.rng,
            ) for n, random_n_loc in zip(nodes, num_edit_locs)]
        fitnesses = self.model.get_fitness(seqs)
        
        return [
            RolloutNode(
                seq=seq,
                num_edits_from_root=n.num_edits_from_root + random_n_loc,
                num_edits_from_parent=random_n_loc,
                fitness=f,
            )
            for seq, n, random_n_loc, f in zip(seqs, nodes, num_edit_locs, fitnesses)
        ]
