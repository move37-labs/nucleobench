"""Common utilities for AdaLead."""

import dataclasses
from typing import Optional, Union

import numpy as np
from scipy.stats import binom
import torch
import random
import xxhash

from nucleobench.optimizations import typing

from nucleobench.optimizations import utils as opt_utils


SequenceType = typing.SequenceType


@dataclasses.dataclass(frozen=True)
class RolloutNode:
    """Class for tracking rollout node.
    
    NOTE on terminology:
    
    a -> b -> c
    
    `a` is the root of `b` and `c`.
    `a` is the parent of `b`.
    `b` is the parent of `c`.
    
    """
    seq: SequenceType
    fitness: np.float32

class ModelWrapper:
    def __init__(self, 
                 model,
                 use_cache: bool = False, 
                 debug: bool = False,
                 ):
        self.model = model
        self.cost = 0
        self.use_cache = use_cache
        self.cache = {}
        self.debug = debug
        
        
    def str_in_cache(self, seq: str) -> bool:
        """Check if a sequence is in the cache."""
        k = xxhash.xxh64(seq).intdigest()
        return k in self.cache

    def get_fitness(self, m_input: list):
        self.cost += len(m_input)
        
        if self.use_cache:
            # 1) Sift sequences into seen and unseen, keeping track of their location
            # so we can preserve order.
            # 2) Pull from the has the fitness of the seen sequences.
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
                    print(f'Cache hit: {len(seen_fitness)}')
                    
        if len(m_input) == 0:
            results = []
        else:
            results = self.model(m_input)
        
        if self.use_cache:
            # 3) Add the unseen sequences to the cache.
            # 4) Interleave seen and unseen results to preserve order.
            for k, v in zip(unseen_hash, results):
                self.cache[k] = v
            unseen_fitness = [(i, r) for (i, _), r in zip(unseen_seq, results)]
            results = [x[1] for x in sorted(seen_fitness + unseen_fitness)]
        
        # Ada* is formulated to maximize fitness, but we want to minimize.
        return [-x for x in results]
    
    
def generate_random_mutant(
    sequence: str, 
    positions_to_mutate: list[str],
    mu: float, 
    alphabet: str, 
    rng: random.Random,
) -> str:
    """
    Generate a mutant of `sequence` where each residue mutates with probability `mu`.

    So the expected value of the total number of mutations is `len(positions_to_mutate) * mu`.
    
    NOTE: This is used in adalead_ref, with rejection sampling. For efficiency, we prefer 
    `generate_random_mutant_v2` since it avoids the need for rejection sampling.

    Args:
        sequence: Sequence that will be mutated from.
        positions_to_mutate: Allowed positions to be mutated.
        mu: Probability of mutation per residue.
        alphabet: Alphabet string.
        rng: Random number generator.

    Returns:
        Mutant sequence string.

    """
    mutant = []
    for i, s in enumerate(sequence):
        if i in positions_to_mutate and rng.random() < mu:
            mutant.append(rng.choice(alphabet))
        else:
            mutant.append(s)
    return "".join(mutant)


def _F_inverse(mu: float, seq_len: int) -> float:
    """F_inverse = 1 - (1-mu')^l """
    return 1 - np.exp( seq_len * np.log1p(-mu) )


def num_edits_likelihood_adalead_legacy(
    num_edits: int,
    seq_len: int,
    mu: float,
    F_inverse: Optional[float] = None,
    ) -> float:
    """The likelihood of `num_edits` edits in the reference Adalead implementation.
    
    Note that the algorithm uses `generate_random_mutant` above, with rejection sampling
    if there are no edits.
    
    See `adalead_utils_test.py` for a test that these are equivalent.
    
    Form:
    mu := mutation rate
    mu' := 3/4 * mu
    l := sequence length
    n := number of edits
    Binom(n, l, mu) := binomial distribution
    
    F := 1 / (1 - (1-mu')^l)
    
    =>
    Pr[N locations edited] = 0, if N <= 0, N > l
    Pr[N locations edited] = Binom(n, l, mu') * F, otherwise
    
    E[num locations edited] = F * mu' * l
        
        
    NOTE: For numerical accuracy, we note the following:
    
    (1 - mu')^l = exp( log( 1 - epsilon)^l ) )
                = exp( l * log( 1 + (-epsilon) ) ) )
                = exp( l * np.log1p(-epsilon) )
    
    """
    if num_edits == 0:
        return 0
    if num_edits < 0 or num_edits > seq_len:
        raise ValueError(f'num_edits must be between 0 and seq_len, inclusive.')
    # Using the notation from above.
    mu_p = 3/4 * mu
    F_inverse = _F_inverse(mu_p, seq_len)
    
    return binom.pmf(num_edits, seq_len, mu_p) / F_inverse


def num_edits_likelihood_adalead_v2(
    num_edits: int,
    seq_len: int,
    mu: float,
    ) -> float:
    """The likelihood of `num_edits` edits.
    
    This doesn't have the 3/4 factor in adalead.
    
    Thus,
    
    E[num locations edited] = F * mu * l
    
    with
    F := 1 / (1 - (1-mu)^l)
    """
    return num_edits_likelihood_adalead_legacy(
        num_edits=num_edits, 
        seq_len=seq_len, 
        mu=mu * 4.0 / 3.0)


def expected_num_edits_adalead_v2(sequence_len: int, mutation_rate: float) -> float:
    F_inverse = _F_inverse(mutation_rate, sequence_len)
    return sequence_len * mutation_rate / F_inverse


def num_edits_likelihood_shifted_binomial(
    num_edits: int,
    seq_len: int,
    mu: float,
    ) -> float:
    """The likelihood of `num_edits` edits, using a shifted binomial distribution.
    
    Modify `mu` so that the expected number of edits is `mu * seq_len`.
    """
    if num_edits == 0: return 0
    return binom.pmf(num_edits - 1, seq_len - 1, mu)


def expected_num_edits_adalead_shifted_binomial(
    sequence_len: int, mutation_rate: float) -> float:
    return 1 + (sequence_len - 1) * mutation_rate


def expected_edits_to_p_shifted_binomial(
    expected_number_of_edits: float,
    sequence_len: int,
):
    """Returns the `p` required for the expected number of edits of the shifted binomial to be `expected_number_of_edits`."""
    if sequence_len <= 1:
        raise ValueError(f'Sequence length must be greater than 1: {sequence_len}')
    if expected_number_of_edits <= 1:
        raise ValueError(f'Expected number of edits must be greater than 1: {expected_number_of_edits}')
    if expected_number_of_edits > sequence_len:
        raise ValueError(f'Expected number of edits must be less than sequence length: {expected_number_of_edits} > {sequence_len}')
    return float(expected_number_of_edits - 1) / (sequence_len - 1)


class NumberEditsSampler(object):
    """Samples the number of edits to make."""
    
    def __init__(
        self, 
        sequence_len: int, 
        mutation_rate: float,
        likelihood_fn: callable,
        rng_seed: int = 0):
        
        self.seq_len = sequence_len
        self.mu = mutation_rate
        self.rng = np.random.default_rng(rng_seed)
        self.num_edits = list(range(1, self.seq_len + 1))
        
        self.probs = [likelihood_fn(n, self.seq_len, self.mu) for n in self.num_edits]
        
        # Do a sample as a sanity check.
        try:
            _ = self.sample(1)
        except ValueError:
            raise ValueError(f'Sum issue: {np.sum(self.probs)} {self.probs}')
        
        
    def expected_num_edits(self) -> float:
        """Returns the expected number of edits."""
        return np.sum(np.array(self.num_edits) * np.array(self.probs))

        
    def sample(self, n_samples: int) -> list[int]:
        return self.rng.choice(self.num_edits, n_samples, p=self.probs)
        
        
def generate_random_mutant_v2(
    sequence: str, 
    positions_to_mutate: list[int],
    random_n_loc: int,
    alphabet: str, 
    rng: np.random.Generator,
) -> str:
    """
    Generate a mutant of `sequence` with exactly `random_n_loc` edits.

    Args:
        sequence: Sequence that will be mutated from.
        positions_to_mutate: Allowed positions to be mutated.
        random_n_loc: Number of mutations per sequence.
        alphabet: Alphabet string.
        rng: Random number generator.

    Returns:
        Mutant sequence string.

    """
    assert isinstance(alphabet, str)
    
    locations_to_edit = opt_utils.get_locations_to_edit(
        positions_to_mutate=positions_to_mutate, 
        random_n_loc=random_n_loc, 
        rng=rng, 
        method='random')
    assert len(locations_to_edit) == random_n_loc
    
    return opt_utils.generate_single_mutant_multiedits(
        base_str=sequence,
        locs_to_edit=locations_to_edit,
        alphabet=list(alphabet),
        rng=rng,
    )
    



def recombine_population(
    gen: list[str],
    rng: random.Random,
    recomb_rate: float,
    positions_to_mutate: list[int],
    ) -> list[str]:
    # If only one member of population, can't do any recombining.
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

            # Put together recombinants.
            if switch:
                strA.append(gen[i][ind])
                strB.append(gen[i + 1][ind])
            else:
                strB.append(gen[i][ind])
                strA.append(gen[i + 1][ind])

        ret.append("".join(strA))
        ret.append("".join(strB))
    return ret


def threshold_on_fitness_percentile(
    in_seqs: list[str], 
    in_seq_scores: np.ndarray, 
    threshold: float,
    debug: bool = False,
    ) -> tuple[list[str], np.ndarray]:
    """Get all sequences within `threshold` percentile of the top_fitness."""
    in_nodes = [RolloutNode(seq, 0, 0, score) for seq, score in zip(in_seqs, in_seq_scores)]
    out_nodes = threshold_nodes_on_fitness_percentile(
        in_nodes=in_nodes, 
        threshold=threshold, 
        debug=debug)
    out_seqs = [node.seq for node in out_nodes]
    out_seq_scores = np.array([node.fitness for node in out_nodes])

    return out_seqs, out_seq_scores


def threshold_nodes_on_fitness_percentile(
    in_nodes: list[RolloutNode], 
    threshold: float,
    debug: bool = False,
    ) -> list[RolloutNode]:
    """Get all sequences within `threshold` percentile of the top_fitness."""
    in_seq_scores = np.array([node.fitness for node in in_nodes])
    in_seqs = [node.seq for node in in_nodes]
    
    top_fitness = in_seq_scores.max()
    parent_mask = in_seq_scores >= top_fitness * (1 - np.sign(top_fitness) * threshold)
    parent_inds = np.argwhere(parent_mask).flatten()
    out_nodes = [in_nodes[i] for i in parent_inds]
    
    if debug:
        print(f'Thresholding went from {len(in_seqs)} to {len(out_nodes)}')
    
    return out_nodes


def softmax(x):
    """
    Computes the softmax function in a numerically stable way.

    Args:
        x: A NumPy array of any shape.

    Returns:
        A NumPy array with the same shape as x, where each element is the softmax of the corresponding row.
    """
    # Subtract the maximum value for numerical stability
    x_shifted = x - np.max(x, keepdims=True)
    
    # Calculate exponentials
    exp_x = np.exp(x_shifted)
    
    # Normalize by the sum of exponentials
    return exp_x / np.sum(exp_x, keepdims=True)


def get_batched_fitness(
    model_wrapper: ModelWrapper,
    sequences: list[str],
    batch_size: int,
) -> np.ndarray:
    """Get fitness for a list of sequences in batches."""
    if len(sequences) == 0:
        return np.array([])

    fitness = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        batch_fitness = model_wrapper.get_fitness(batch)
        fitness.extend(batch_fitness)
    
    return np.array(fitness)