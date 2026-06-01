"""GradaBeamPBT.

Gradient-guided adaptive beam, adaptive mutation rate (PBT), adaptive directed evolution.
"""

import argparse
import dataclasses
from dataclasses import field
from functools import cache  # Added for PBT sampler caching

import numpy as np
from scipy.special import softmax

from nucleobench.common import argparse_lib, constants, testing_utils
from nucleobench.optimizations import optimization_class as oc
from nucleobench.optimizations.typing import ModelType, SamplesType, SequenceType

from .. import ada_utils

PositionsAndCharactersType = ada_utils.PositionsAndCharactersType


@dataclasses.dataclass(frozen=True)
class RolloutNodeWithProbs(ada_utils.RolloutNode):
    """Class for tracking rollout node with probabilities."""

    probs: np.ndarray | None = field(default=None, hash=False, compare=False)
    pos_and_chars: PositionsAndCharactersType | None = field(
        default=None, hash=False, compare=False
    )
    edits_since_root: int | None = None
    # [PBT Modification]:
    mutations_per_sequence: float = dataclasses.field(
        default=1.0, compare=False, hash=True
    )
    exploration_alpha: float = dataclasses.field(default=0.05, compare=False, hash=True)


RolloutNode = RolloutNodeWithProbs


class GradaBeam(oc.SequenceOptimizer):
    """GradaBeam nucleic acid sequence designer with PBT."""

    def __init__(
        self,
        model_fn: ModelType,
        start_sequence: SequenceType,
        mutations_per_sequence: float,
        beam_size: int,
        n_rollouts_per_root: int,
        exploration_alpha: float,
        max_rollout_len: int = 200,
        gradient_prob_cap: float = 0.10,
        max_logit: float = 3.0,
        rng_seed: int = 0,
        positions_to_mutate: list[int] | None = None,
        eval_batch_size: int = 1,
        use_pbt: bool = True,
        debug: bool = False,
    ):
        self.positions_to_mutate = positions_to_mutate or list(
            range(len(start_sequence))
        )
        self.tism_positions = (
            None
            if len(self.positions_to_mutate) == len(start_sequence)
            else self.positions_to_mutate
        )

        assert min(self.positions_to_mutate) >= 0
        assert max(self.positions_to_mutate) < len(start_sequence)
        assert mutations_per_sequence > 0
        assert beam_size > 0
        assert n_rollouts_per_root > 0
        assert exploration_alpha >= 0 and exploration_alpha <= 1

        self.exploration_alpha = exploration_alpha
        self.use_pbt = use_pbt

        self.model = ada_utils.ModelWrapper(
            model_fn,
            use_cache=True,
            debug=debug,
            tism_cost=1.0,
            start_sequence=start_sequence,
        )
        self.start_sequence = start_sequence
        self.beam_size = beam_size
        self.n_rollouts_per_root = n_rollouts_per_root
        self.alphabet = "".join(constants.VOCAB)
        self.positions_to_mutate_set = set(self.positions_to_mutate)
        self.eval_batch_size = eval_batch_size
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)

        self.max_rollout_len = max_rollout_len
        self.gradient_prob_cap = gradient_prob_cap
        self.max_logit = max_logit
        self.debug = debug

        assert isinstance(start_sequence, str)
        seed_node = RolloutNode(
            seq=start_sequence,
            fitness=np.float32(0.0),
            edits_since_root=0,
            probs=None,
            pos_and_chars=None,
            mutations_per_sequence=float(mutations_per_sequence),  # [PBT Modification]
            exploration_alpha=float(exploration_alpha),  # [PBT Modification]
        )

        # Initialize with gradient-based mutations
        initialized_roots = self.initialize_roots_with_gradients(
            [seed_node] * beam_size
        )

        # [PBT Modification]: Setup initial PBT sampling
        initial_sampler = self.get_sampler(seed_node.mutations_per_sequence)
        num_edit_locs = [int(x) for x in initial_sampler.sample(beam_size)]

        self.current_nodes = []
        for i in range(0, beam_size, self.eval_batch_size):
            cur_num_edits = num_edit_locs[i : i + self.eval_batch_size]
            cur_roots = initialized_roots[i : i + self.eval_batch_size]
            self.current_nodes.extend(
                self.mutate_nodes_gradabeam(
                    cur_roots,
                    cur_num_edits,
                    [seed_node.mutations_per_sequence] * len(cur_num_edits),
                )
            )

    # [PBT Modification]: Added helper
    def get_sampler(
        self, mutations_per_sequence: float
    ) -> ada_utils.NumberEditsSampler:
        rounded_rate = round(mutations_per_sequence, 4)
        return self._get_sampler_cached(rounded_rate)

    # [PBT Modification]: Added helper
    @cache
    def _get_sampler_cached(
        self, mutations_per_sequence: float
    ) -> ada_utils.NumberEditsSampler:
        mu = mutations_per_sequence / len(self.positions_to_mutate)
        return ada_utils.NumberEditsSamplerAdaBeam(
            sequence_len=len(self.positions_to_mutate),
            mutation_rate=mu,
            rng_seed=self.rng_seed,
        )

    # [PBT Modification]: Added helper
    def _get_next_mutation_params(self, node: RolloutNode) -> tuple[int, float]:
        """Calculates n_edits, new mutation rate, and target alpha for the child node."""
        current_rate = node.mutations_per_sequence

        if self.use_pbt:
            # Direct snap mode: sample edits using current rate, then snap rate to observation
            n_edits = int(self.get_sampler(current_rate).sample(1)[0])
            new_rate = float(max(1.0, n_edits))

            return n_edits, new_rate
        else:
            # PBT disabled: keep everything constant
            n_edits = int(self.get_sampler(current_rate).sample(1)[0])
            return n_edits, current_rate

    def get_batched_fitness(self, sequences: list[str]) -> np.ndarray:
        return ada_utils.get_batched_fitness(
            model_wrapper=self.model,
            sequences=sequences,
            batch_size=self.eval_batch_size,
        )

    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        group = parser.add_argument_group("GradaBeam init args")
        # ... [Existing args] ...
        group.add_argument("--beam_size", type=int, default=10, required=False)
        group.add_argument("--mutations_per_sequence", type=float, required=False)
        group.add_argument("--n_rollouts_per_root", type=int, default=4, required=False)
        group.add_argument("--eval_batch_size", type=int, default=1, required=False)
        group.add_argument("--rng_seed", type=int, default=42, required=False)
        group.add_argument("--exploration_alpha", type=float, required=True)
        group.add_argument("--max_rollout_len", type=int, default=200, required=False)
        group.add_argument(
            "--use_pbt",
            type=argparse_lib.str_to_bool,
            default=True,
            required=False,
            help="Enable Population Based Training for adaptive mutation rates.",
        )
        group.add_argument(
            "--debug", type=argparse_lib.str_to_bool, default=None, required=False
        )
        return parser

    @staticmethod
    def debug_init_args():
        return {
            "model_fn": testing_utils.CountLetterModel(flip_sign=True),
            "start_sequence": "AAAAAA",
            "beam_size": 10,
            "mutations_per_sequence": 1,
            "n_rollouts_per_root": 4,
            "eval_batch_size": 1,
            "rng_seed": 42,
            "exploration_alpha": 0.05,
            "use_pbt": True,
        }

    def run(self, n_steps: int):
        for _step in range(n_steps):
            self.current_nodes = self.propose_sequences(self.current_nodes)
            if self.debug and len(self.current_nodes) > 0:
                print(f"Step {_step} top score: {self.current_nodes[0].fitness}")
                rates = [n.mutations_per_sequence for n in self.current_nodes]
                print(f"[PBT] Mutation Rates of top candidates: {rates}")
                alphas = [n.exploration_alpha for n in self.current_nodes]
                print(
                    f"[PBT] Exploration Alphas of top candidates (high is uniform): {alphas}"
                )

    def get_samples(self, n_samples: int) -> SamplesType:
        """Get samples."""
        limit = min(n_samples, len(self.current_nodes))
        sorted_nodes = sorted(self.current_nodes, key=lambda x: x.fitness, reverse=True)
        return [x.seq for x in sorted_nodes][:limit]

    def is_finished(self) -> bool:
        return False

    def propose_sequences(self, root_nodes: list[RolloutNode]) -> list[RolloutNode]:
        """Propose top `beam_size` sequences for evaluation."""
        nodes_visited, rollout_lengths = set(), []
        gradient_node_cache = {}

        root_nodes_effective = root_nodes * self.n_rollouts_per_root
        for i in range(0, len(root_nodes_effective), self.eval_batch_size):
            cur_root_nodes = root_nodes_effective[i : i + self.eval_batch_size]
            parent_nodes = cur_root_nodes

            assert len(parent_nodes) == 1, (
                "GradaBeam propose_sequences expects exactly one parent node."
            )
            parent_seq = parent_nodes[0].seq

            if parent_seq in gradient_node_cache:
                parent_nodes = [gradient_node_cache[parent_seq]]
            else:
                parent_nodes = self.initialize_roots_with_gradients(parent_nodes)
                gradient_node_cache[parent_seq] = parent_nodes[0]

            cur_nodes_visited, rollout_lengths = self.rollout(parent_nodes=parent_nodes)
            nodes_visited.update(cur_nodes_visited)
            rollout_lengths.extend(rollout_lengths)

        if len(nodes_visited) == 0:
            raise ValueError("No nodes generated.")

        nodes_visited = sorted(nodes_visited, key=lambda x: x.fitness, reverse=True)
        top_nodes = nodes_visited[: self.beam_size]

        return top_nodes

    def initialize_roots_with_gradients(
        self, nodes: list[RolloutNode]
    ) -> list[RolloutNode]:
        """Calculates gradients for roots and upgrades them to GradientRolloutNodes."""
        probs_list, pos_and_chars_list = self.probabilities_over_actions_from_tism(
            nodes
        )

        grad_nodes = []
        for node, probs, pos_and_chars in zip(nodes, probs_list, pos_and_chars_list):
            grad_nodes.append(
                RolloutNode(
                    seq=node.seq,
                    fitness=node.fitness,
                    edits_since_root=0,
                    probs=probs,
                    pos_and_chars=pos_and_chars,
                    mutations_per_sequence=node.mutations_per_sequence,
                    exploration_alpha=node.exploration_alpha,
                )
            )
        return grad_nodes

    def rollout(
        self, parent_nodes: list[RolloutNode]
    ) -> tuple[set[RolloutNode], list[int]]:
        """Rollout with PBT."""
        nodes_visited, rollout_lengths = set(), []

        cur_rollout_length = 0
        while len(parent_nodes) > 0 and cur_rollout_length < self.max_rollout_len:
            # [PBT Modification]: Calculate dynamic rates/edits/alpha per parent
            num_edit_locs, new_rates = [], []
            for n in parent_nodes:
                n_edits, new_rate = self._get_next_mutation_params(n)
                num_edit_locs.append(n_edits)
                new_rates.append(new_rate)

            # [PBT Modification]: Pass new rates and target alphas to mutate
            children = self.mutate_nodes_gradabeam(
                parent_nodes, num_edit_locs, new_rates
            )

            nodes_visited.update(children)

            cur_rollout_length += 1
            new_nodes = []
            for child, comparison_node in zip(children, parent_nodes):
                if child.fitness >= comparison_node.fitness:
                    new_nodes.append(child)
                else:
                    rollout_lengths.append(cur_rollout_length)
            parent_nodes = new_nodes

        return nodes_visited, rollout_lengths

    def mutate_nodes_gradabeam(
        self,
        nodes: list[RolloutNode],
        num_edit_locs: list[int],
        new_rates: list[float],
    ) -> list[RolloutNode]:

        # [PBT Modification]: Validation
        assert (
            len(nodes) == len(num_edit_locs) == len(new_rates) <= self.eval_batch_size
        )

        seqs, new_probs, num_edits_effective, child_alphas = [], [], [], []
        for node, num_edits in zip(nodes, num_edit_locs):
            num_available = (node.probs > 0).sum()
            effective_num_edits = min(num_edits, num_available)
            assert effective_num_edits > 0
            num_edits_effective.append(effective_num_edits)

            candidate, rel_pos_of_mutations = ada_utils.generate_random_mutant_tism(
                sequence=node.seq,
                pos_and_chars_to_mutate=node.pos_and_chars,
                random_n_loc=effective_num_edits,
                rng=self.rng,
                probs=node.probs,
                debug=self.debug,
            )
            seqs.append(candidate)

            # --- MASKING LOGIC ---
            # Zero out the positions we just changed
            # (We cannot trust the old gradient at these new chars)
            child_probs = node.probs.copy()
            child_probs[rel_pos_of_mutations] = 0.0
            total_p = child_probs.sum()
            if total_p > 0:
                child_probs /= total_p
            else:
                # Fallback: If we exhausted all probability mass,
                # revert to uniform or stop mutating.
                child_probs = np.ones_like(child_probs) / len(child_probs)
            new_probs.append(child_probs)

            # --- DIRECT SNAP FOR ALPHA ---
            if self.use_pbt:
                # Calculate posterior probability that mutations were uniform
                p_uniform = 1.0 / len(node.probs)
                # Get the P_final (from node.probs) for the chosen indices
                P_final_values = node.probs[rel_pos_of_mutations]
                # Calculate posterior per mutation: P(uniform | observed) = (alpha * p_uniform) / P_final
                # Add small epsilon to avoid division by zero
                posteriors = (node.exploration_alpha * p_uniform) / (
                    P_final_values + 1e-10
                )
                # Average this posterior over the number of edits made
                avg_posterior = float(np.mean(posteriors))
                child_alpha = float(np.clip(avg_posterior, 0.01, 0.99))
            else:
                # PBT disabled: keep alpha constant
                child_alpha = node.exploration_alpha

            child_alphas.append(child_alpha)

        fitnesses = self.get_batched_fitness(seqs)

        return [
            RolloutNode(
                seq=seq,
                fitness=float(f),
                probs=probs,
                edits_since_root=n.edits_since_root + int(num_edits),
                pos_and_chars=n.pos_and_chars,
                # [PBT Modification]: Child inherits new rate and alpha
                mutations_per_sequence=new_rate,
                exploration_alpha=child_alpha,
            )
            for seq, f, probs, n, num_edits, new_rate, child_alpha in zip(
                seqs,
                fitnesses,
                new_probs,
                nodes,
                num_edits_effective,
                new_rates,
                child_alphas,
            )
        ]

    # ... [Rest of file: probabilities_over_actions_from_tism, logits_to_probs] ...
    # (Functions below can remain identical to the original)
    def probabilities_over_actions_from_tism(
        self, nodes: list[RolloutNode]
    ) -> tuple[list[float], list[PositionsAndCharactersType]]:
        # ... [Same as original] ...
        probs_list, pos_and_chars_list = [], []
        for n in nodes:
            pos_and_chars, logits = self.model.get_tism(
                sequence=n.seq, idxs=self.tism_positions, debug=self.debug
            )
            # Make sure `pos_to_mutate` is respected.
            assert len(pos_and_chars) == 3 * len(self.positions_to_mutate), (
                len(pos_and_chars),
                len(self.positions_to_mutate),
                self.tism_positions,
            )
            assert len(pos_and_chars) == len(logits)

            # 2. Compute Probabilities
            # This handles Temperature, Stability, and Exploration in one step.
            probs = self.logits_to_probs(logits, n.exploration_alpha)
            probs_list.append(probs)
            pos_and_chars_list.append(pos_and_chars)
        return probs_list, pos_and_chars_list

    def logits_to_probs(self, logits: np.ndarray, alpha: float) -> np.ndarray:
        # Normalize logit standard deviation.
        std_dev = np.std(logits)
        if std_dev < 1e-9:
            return np.ones_like(logits) / len(logits)
        scaled_logits = logits / std_dev

        # Scale logits by a dynamic temperature.
        dynamic_temp = max(1.0, np.max(scaled_logits) / self.max_logit)
        scaled_logits = scaled_logits / dynamic_temp

        gradient_probs = softmax(scaled_logits)

        # Somewhat normalize probabilities to a cap.
        gradient_probs = np.minimum(gradient_probs, self.gradient_prob_cap)
        gradient_probs /= np.sum(gradient_probs)

        # Mix in uniform exploration.
        n_actions = len(scaled_logits)
        uniform_probs = np.ones(n_actions) / n_actions
        final_probs = ((1.0 - alpha) * gradient_probs) + (alpha * uniform_probs)

        return final_probs / np.sum(final_probs)
