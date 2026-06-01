"""Common utilities for [Gr]Ada*."""

import dataclasses
import random

import numpy as np
import torch
import xxhash
from scipy.stats import binom

from nucleobench.optimizations import typing
from nucleobench.optimizations import utils as opt_utils

SequenceType = typing.SequenceType
TISMType = typing.TISMType
PositionsAndCharactersType = list[tuple[int, str]]
LogitsType = np.ndarray


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
                 model: typing.ModelType,
                 use_cache: bool = False,
                 cache_limit: int = 100000,
                 debug: bool = False,
                 tism_cost: float | None = None,
                 start_sequence: str | None = None,
                 ):
        if tism_cost is not None:
            assert hasattr(model, 'tism_torch'), \
                "Model must have tism_torch method. This is required for optimized get_tisms."
        self.model = model
        self.cost = 0
        self.use_cache = use_cache
        self.cache_limit = cache_limit
        self.cache = {}
        self.debug = debug
        self.tism_cost = tism_cost

        # Double check that the model is in evaluation mode.
        # TODO(joelshor): Force this to happen if the model is a PyTorch model.
        try:
            self.model.eval()
        except AttributeError:
            try:
                self.model.model.eval()
            except:
                pass

        if self.tism_cost is not None:
            # Some optimizations for backprop:
            # We only need gradients for the input, so disable the rest.
            try:
                for param in self.model.parameters():
                    param.requires_grad = False
            except AttributeError:
                for param in self.model.model.parameters(): # Access the underlying torch module
                    param.requires_grad = False

        # The above is stochastic. Work around it.
        del start_sequence  # Unused.
        if 'Rinalmo' in type(self.model).__name__:
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
            # SAFETY VALVE: Prevent infinite growth for long runs
            if len(self.cache) > self.cache_limit:
                if self.debug: print("Cache limit reached. Flushing.")
                self.cache = {}

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
            # `torch.inference_mode()` is faster than `torch.no_grad()`, but
            # doesn't work with RinAlmo's jit.compile optimization,
            # so we use the fastest we can.
            with self.torch_opt_fn():
                results = self.model(m_input)

        if self.use_cache:
            # 3) Add the unseen sequences to the cache.
            # 4) Interleave seen and unseen results to preserve order.
            for k, v in zip(unseen_hash, results):
                self.cache[k] = v
            unseen_fitness = [(i, r) for (i, _), r in zip(unseen_seq, results)]
            results = [x[1] for x in sorted(seen_fitness + unseen_fitness)]

        # Ada* is formulated to maximize fitness, but we want to minimize.
        return [-float(x) for x in results]


    def get_tism(
        self,
        sequence: str,
        idxs: list[int] | None = None,
        debug: bool = False,
        ) -> tuple[PositionsAndCharactersType, LogitsType]:
        del debug  # Unused.
        assert hasattr(self.model, 'tism_torch'), \
            "Model must have tism_torch method. This is required for optimized get_tisms."

        if self.tism_cost is None:
            raise ValueError('Cost can\'t be None.')
        if self.tism_cost < 1.0:
            raise ValueError('Cost must be >= 1.0.')
        self.cost += self.tism_cost

        # Use fast tensor-based TISM
        pos_and_chars_to_mutate, logits = self.model.get_tism(sequence, idxs)

        logits *= -1  # Flip the sign, to conform to convention.

        return (pos_and_chars_to_mutate, logits)


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
    return -np.expm1( seq_len * np.log1p(-mu) )


def num_edits_likelihood_adalead_legacy(
    num_edits: int,
    seq_len: int,
    mu: float,
    F_inverse: float | None = None,
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
    return num_edits_likelihood_adabeam(
        num_edits=num_edits,
        seq_len=seq_len,
        mu=mu * 3.0 / 4.0)


def num_edits_likelihood_adabeam(
    num_edits: np.ndarray,
    seq_len: int,
    mu: float,
    ) -> float:
    """The likelihood of `num_edits` edits in the reference AdaBeam implementation.
    
    Thus,
    
    E[num locations edited] = F * mu * l
    
    Form:
    mu := mutation rate
    l := sequence length
    n := number of edits
    Binom(n, l, mu) := binomial distribution
    
    with
    F := 1 / (1 - (1-mu)^l)
    
    =>
    Pr[N locations edited] = 0, if N <= 0, N > l
    Pr[N locations edited] = Binom(n, l, mu) * F, otherwise
    
    E[num locations edited] = F * mu * l
        
        
    NOTE: For numerical accuracy, we note the following:
    
    (1 - mu')^l = exp( log( 1 - epsilon)^l ) )
                = exp( l * log( 1 + (-epsilon) ) ) )
                = exp( l * np.log1p(-epsilon) )
    """
    assert isinstance(num_edits, np.ndarray)
    if num_edits.min() < 0 or num_edits.max() > seq_len:
        raise ValueError('num_edits must be between 0 and seq_len, inclusive.')

    # Using the notation from above.
    F_inverse = _F_inverse(mu, seq_len)

    probs = binom.pmf(num_edits, seq_len, mu) / F_inverse

    # The Binomial distribution has support at k=0, but AdaBeam defines P(0)=0.
    # We force any element where num_edits == 0 to have probability 0.0.
    probs[num_edits == 0] = 0.0

    return probs


def expected_num_edits_adalead_v2(sequence_len: int, mutation_rate: float) -> float:
    F_inverse = _F_inverse(mutation_rate, sequence_len)
    return sequence_len * mutation_rate / F_inverse


class NumberEditsSampler:
    """Vectorized samples the number of edits to make."""

    def __init__(
        self,
        sequence_len: int,
        mutation_rate: float,
        likelihood_fn: callable,
        rng_seed: int = 0):

        self.seq_len = sequence_len
        self.mu = mutation_rate
        self.rng = np.random.default_rng(rng_seed)

        self.num_edits = np.arange(1, self.seq_len + 1, dtype=np.uint32)

        self.probs = likelihood_fn(self.num_edits, self.seq_len, self.mu)


    def expected_num_edits(self) -> float:
        """Returns the expected number of edits."""
        return np.sum(self.num_edits * self.probs)


    def sample(self, n_samples: int) -> list[int]:
        # OPTIMIZATION: Use numpy array directly - faster than converting from list.
        return self.rng.choice(self.num_edits, size=n_samples, p=self.probs)


class NumberEditsSamplerAdaBeam(NumberEditsSampler):
    """Samples the number of edits to make."""

    def __init__(
        self,
        sequence_len: int,
        mutation_rate: float,
        rng_seed: int = 0):

        super().__init__(
            sequence_len=sequence_len,
            mutation_rate=mutation_rate,
            rng_seed=rng_seed,
            likelihood_fn=num_edits_likelihood_adabeam,
        )


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


def generate_random_mutant_tism(
    sequence: str,
    pos_and_chars_to_mutate: PositionsAndCharactersType,
    random_n_loc: int,
    rng: np.random.Generator,
    probs: np.ndarray,
    debug: bool = False,
) -> tuple[str, list[int]]:
    """
    Generate a mutant of `sequence` with exactly `random_n_loc` edits, using `tism` info.

    Args:
        sequence: Sequence that will be mutated from.
        pos_and_chars_to_mutate: (position, character) of the allowed positions to be mutated.
        random_n_loc: Number of mutations per sequence.
        alphabet: Alphabet string.
        rng: Random number generator.
        probs: XXX
        debug: If True, print debug info.

    Returns:
        Mutant sequence string and indices within the mutable positions that were mutated.

    """
    assert isinstance(pos_and_chars_to_mutate, list)

    # OPTIMIZATION: Use integer indices instead of tuples for faster rng.choice
    # NumPy's rng.choice is much faster when working with integer arrays
    n_actions = len(pos_and_chars_to_mutate)
    indices = np.arange(n_actions, dtype=np.uint32)

    selected_indices = rng.choice(
        indices,
        size=random_n_loc,
        replace=False,
        p=probs)
    assert len(selected_indices) == random_n_loc

    mutant, rel_pos_of_mutations = list(sequence), []
    for i in selected_indices:
        pos, char = pos_and_chars_to_mutate[i]
        mutant[int(pos)] = str(char)
        rel_pos_of_mutations.append(i)  # Use relative position, which is needed downstream.
    return ''.join(mutant), rel_pos_of_mutations


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
        assert isinstance(batch_fitness, list)
        for x in batch_fitness:
            assert isinstance(x, float), (type(x), x)
        fitness.extend(batch_fitness)

    return np.array(fitness)
