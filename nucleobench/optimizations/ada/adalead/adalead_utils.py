"""Common utilities for AdaLead."""

import dataclasses
import random

import numpy as np
import torch
import xxhash
from scipy.stats import binom

from nucleobench.optimizations import typing

SequenceType = typing.SequenceType
PositionsAndCharactersType = list[tuple[int, str]]
LogitsType = np.ndarray


@dataclasses.dataclass(frozen=True)
class RolloutNode:
    """Class for tracking rollout node."""

    seq: SequenceType
    fitness: np.float32


class ModelWrapper:
    def __init__(
        self,
        model: typing.ModelType,
        use_cache: bool = False,
        cache_limit: int = 100000,
        debug: bool = False,
        tism_cost: float | None = None,
        start_sequence: str | None = None,
    ):
        if tism_cost is not None:
            assert hasattr(model, "tism_torch"), (
                "Model must have tism_torch method. This is required for optimized get_tisms."
            )
        self.model = model
        self.cost = 0
        self.use_cache = use_cache
        self.cache_limit = cache_limit
        self.cache = {}
        self.debug = debug
        self.tism_cost = tism_cost

        # Double check that the model is in evaluation mode.
        try:
            self.model.eval()
        except AttributeError:
            try:
                self.model.model.eval()
            except Exception:
                pass

        if self.tism_cost is not None:
            try:
                for param in self.model.parameters():
                    param.requires_grad = False
            except AttributeError:
                for param in (
                    self.model.model.parameters()
                ):  # Access the underlying torch module
                    param.requires_grad = False

        del start_sequence  # Unused.
        if "Rinalmo" in type(self.model).__name__:
            self.torch_opt_fn = torch.no_grad
        else:
            self.torch_opt_fn = torch.inference_mode

    def str_in_cache(self, seq: str) -> bool:
        """Check if a sequence is in the cache."""
        k = xxhash.xxh64(seq).intdigest()
        return k in self.cache

    def get_fitness(self, m_input: list) -> list[float]:
        self.cost += len(m_input)

        if self.use_cache:
            if len(self.cache) > self.cache_limit:
                if self.debug:
                    print("Cache limit reached. Flushing.")
                self.cache = {}

            seen_fitness, unseen_seq, unseen_hash = [], [], []
            for i, seq in enumerate(m_input):
                k = xxhash.xxh64(seq).intdigest()
                if k in self.cache:
                    seen_fitness.append((i, self.cache[k]))
                else:
                    unseen_seq.append((i, seq))
                    unseen_hash.append(k)
            m_input = [seq for _, seq in unseen_seq]

            if self.debug:
                if len(seen_fitness) > 0:
                    print(f"Cache hit: {len(seen_fitness)}")

        if len(m_input) == 0:
            results = []
        else:
            with self.torch_opt_fn():
                results = self.model(m_input)

        if self.use_cache:
            for k, v in zip(unseen_hash, results):
                self.cache[k] = v
            unseen_fitness = [(i, r) for (i, _), r in zip(unseen_seq, results)]
            results = [x[1] for x in sorted(seen_fitness + unseen_fitness)]

        return [-float(x) for x in results]

    def get_tism(
        self,
        sequence: str,
        idxs: list[int] | None = None,
        debug: bool = False,
    ) -> tuple[PositionsAndCharactersType, LogitsType]:
        del debug  # Unused.
        assert hasattr(self.model, "tism_torch"), (
            "Model must have tism_torch method. This is required for optimized get_tisms."
        )

        if self.tism_cost is None:
            raise ValueError("Cost can't be None.")
        if self.tism_cost < 1.0:
            raise ValueError("Cost must be >= 1.0.")
        self.cost += self.tism_cost

        pos_and_chars_to_mutate, logits = self.model.get_tism(sequence, idxs)
        logits *= -1  # Flip the sign, to conform to convention.
        return (pos_and_chars_to_mutate, logits)


def generate_random_mutant(
    sequence: str,
    positions_to_mutate: list[str] | list[int],
    mu: float,
    alphabet: str,
    rng: random.Random,
) -> str:
    """Generate a mutant of `sequence` where each residue mutates with probability `mu`."""
    mutant = []
    for i, s in enumerate(sequence):
        if i in positions_to_mutate and rng.random() < mu:
            mutant.append(rng.choice(alphabet))
        else:
            mutant.append(s)
    return "".join(mutant)


def _F_inverse(mu: float, seq_len: int) -> float:
    """F_inverse = 1 - (1-mu')^l"""
    return -np.expm1(seq_len * np.log1p(-mu))


def num_edits_likelihood_adabeam(
    num_edits: np.ndarray,
    seq_len: int,
    mu: float,
) -> float:
    """The likelihood of `num_edits` edits in the reference AdaBeam implementation."""
    assert isinstance(num_edits, np.ndarray)
    if num_edits.min() < 0 or num_edits.max() > seq_len:
        raise ValueError("num_edits must be between 0 and seq_len, inclusive.")

    F_inverse = _F_inverse(mu, seq_len)
    probs = binom.pmf(num_edits, seq_len, mu) / F_inverse
    probs[num_edits == 0] = 0.0
    return probs


def num_edits_likelihood_adalead_legacy(
    num_edits: int | np.ndarray,
    seq_len: int,
    mu: float,
    F_inverse: float | None = None,
) -> float:
    """The likelihood of `num_edits` edits in the reference Adalead implementation."""
    if isinstance(num_edits, int):
        num_edits = np.array([num_edits])
    return num_edits_likelihood_adabeam(
        num_edits=num_edits, seq_len=seq_len, mu=mu * 3.0 / 4.0
    )


def expected_num_edits_adalead_v2(sequence_len: int, mutation_rate: float) -> float:
    F_inverse = _F_inverse(mutation_rate, sequence_len)
    return sequence_len * mutation_rate / F_inverse


def recombine_population(
    gen: list[str],
    rng: random.Random,
    recomb_rate: float,
    positions_to_mutate: list[int],
) -> list[str]:
    if len(gen) == 1:
        return gen

    rng.shuffle(gen)
    ret = []
    for i in range(0, len(gen) - 1, 2):
        strA = []
        strB = []
        switch = False
        for ind in positions_to_mutate:
            if rng.random() < recomb_rate:
                switch = not switch

            if switch:
                strA.append(gen[i][ind])
                strB.append(gen[i + 1][ind])
            else:
                strB.append(gen[i][ind])
                strA.append(gen[i + 1][ind])

        ret.append("".join(strA))
        ret.append("".join(strB))
    return ret


def threshold_nodes_on_fitness_percentile(
    in_nodes: list[RolloutNode],
    threshold: float,
    debug: bool = False,
) -> list[RolloutNode]:
    """Get all sequences within `threshold` percentile of the top_fitness."""
    in_seq_scores = np.array([node.fitness for node in in_nodes])

    top_fitness = in_seq_scores.max()
    parent_mask = in_seq_scores >= top_fitness * (1 - np.sign(top_fitness) * threshold)
    parent_inds = np.argwhere(parent_mask).flatten()
    out_nodes = [in_nodes[i] for i in parent_inds]

    if debug:
        print(f"Thresholding went from {len(in_nodes)} to {len(out_nodes)}")

    return out_nodes
