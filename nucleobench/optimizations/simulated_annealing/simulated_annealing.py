from nucleobench.common import testing_utils
from nucleobench.common import constants
from typing import Callable, Optional
import argparse
import numpy as np
import random

from nucleobench.optimizations import optimization_class as oc

from nucleobench.optimizations.typing import ModelType, PositionsToMutateType, SequenceType, SamplesType


class SimulatedAnnealingBase:
    def __init__(
        self,
        model_fn: Callable,
        start_sequence: SequenceType,
        proposal_fn: Callable,
        temperature_fn: Callable,
        rng_seed: int,
    ):
        self.model_fn = model_fn
        self.proposal_fn = proposal_fn
        self.temperature_fn = temperature_fn

        self.n_steps = 0

        # Simulated annealing doesn't support batch mode for now.
        assert isinstance(start_sequence, str)
        self.start_sequence = start_sequence
        self.seed_score = model_fn([start_sequence])[0]

        self.current_sequence = start_sequence
        self.current_score = self.seed_score

        self.best_sequence = start_sequence
        self.best_score = self.current_score

        self.rng = random.Random(rng_seed)

    def get_samples(self, n_samples: int) -> SamplesType:
        """Get samples."""
        assert n_samples == 1
        return [self.best_sequence]

    def step(self) -> bool:
        """Take a single simulated annealing step."""
        new_sequence = self.proposal_fn(self.current_sequence, self.rng)
        new_score = self.model_fn([new_sequence])[0]
        temperature = self.temperature_fn(self.n_steps)
        delta_score = new_score - self.current_score

        # We always accept proposals that reduce the energy. Proposals that increase the energy are
        # accepted with a probability that depends on the temperature and the energy difference.
        if delta_score < 0:
            acceptance_prob = 1.0
            accepted = True
        else:
            acceptance_prob = np.exp(-delta_score / temperature)
            accepted = self.rng.random() < acceptance_prob

        # if self.n_steps % 100 == 0:
        #     print(
        #         f"delta_score: {delta_score}, temperature: {temperature}, acceptance prob: {acceptance_prob}, accepted: {accepted}"
        #     )
        if accepted:
            self.current_sequence = new_sequence
            self.current_score = new_score
            if new_score < self.best_score:
                self.best_sequence = new_sequence
                self.best_score = new_score

        self.n_steps += 1
        return accepted


class PolynomialDecay:
    def __init__(
        self,
        a=1.0,
        b=1.0,
        p=1.0,
    ):
        self.a = a
        self.b = b
        self.p = p

    def __call__(self, n):
        return self.a * ((1.0 + n / self.b) ** -self.p)


class UniformProposal:
    def __init__(
        self,
        alphabet: str,
        n_mutations: int,
        positions_to_mutate: Optional[list[int]],
    ):
        self.alphabet = alphabet
        self.n_mutations = n_mutations
        self.positions_to_mutate = positions_to_mutate

    def __call__(self, sequence: str, rng: random.Random) -> str:
        for _ in range(self.n_mutations):
            if self.positions_to_mutate is not None:
                pos = rng.choice(self.positions_to_mutate)
            else:
                pos = rng.randint(0, len(sequence) - 1)
            new_char = rng.choice(self.alphabet)
            sequence = sequence[:pos] + new_char + sequence[pos + 1 :]

        return sequence


class SimulatedAnnealing(oc.SequenceOptimizer):
    """Simulated annealing to minimize model_fn on sequences."""

    def __init__(
        self,
        model_fn: ModelType,
        start_sequence: SequenceType,
        polynomial_decay_a: float,
        polynomial_decay_b: float,
        polynomial_decay_p: float,
        n_mutations_per_proposal: int,
        rng_seed: int,
        positions_to_mutate: Optional[PositionsToMutateType] = None,
    ):
        proposal_fn = UniformProposal(
            "".join(constants.VOCAB), n_mutations_per_proposal, positions_to_mutate
        )
        temperature_fn = PolynomialDecay(
            a=polynomial_decay_a, b=polynomial_decay_b, p=polynomial_decay_p
        )

        self.sa = SimulatedAnnealingBase(
            model_fn=model_fn,
            start_sequence=start_sequence,
            proposal_fn=proposal_fn,
            temperature_fn=temperature_fn,
            rng_seed=rng_seed,
        )

    @staticmethod
    def run_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        return parser

    @staticmethod
    def debug_run_args():
        return {}

    def get_samples(self, n_samples: int) -> SamplesType:
        return self.sa.get_samples(n_samples)

    def step(self) -> bool:
        return self.sa.step()

    def run(
        self,
        n_steps: int,
    ):
        for _step in range(n_steps):
            self.step()

    def is_finished(self) -> bool:
        return False
    
    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        group = parser.add_argument_group("Simulated annealing init args")

        group.add_argument(
            "--polynomial_decay_a",
            type=float,
            default=1.0,
            required=False,
            help="Polynomial decay a parameter",
        )
        group.add_argument(
            "--polynomial_decay_b",
            type=float,
            default=1.0,
            required=False,
            help="Polynomial decay b parameter",
        )
        group.add_argument(
            "--polynomial_decay_p",
            type=float,
            default=1.0,
            required=False,
            help="Polynomial decay p parameter",
        )
        group.add_argument(
            "--n_mutations_per_proposal",
            type=int,
            default=1,
            required=False,
            help="Number of mutations per proposal",
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
            "model_fn": testing_utils.CountLetterModel(flip_sign=True),
            "start_sequence": "AAAAAA",
            "positions_to_mutate": None,
            "polynomial_decay_a": 1.0,
            "polynomial_decay_b": 1.0,
            "polynomial_decay_p": 0.5,
            "n_mutations_per_proposal": 1,
            "rng_seed": 42,
        }
