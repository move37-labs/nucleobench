"""Directed evolution, based on gRelu.

Copied and modified from `https://github.com/Genentech/gReLU/blob/main/src/grelu/design.py`,
as per its MIT license.
"""

from typing import Optional

import dataclasses
import random
import tqdm

import numpy as np

from nucleobench.optimizations import model_class as mc
from nucleobench.common import constants


@dataclasses.dataclass
class TISMArgs:
    # If `True`: Use TISM to identify the location but not the mutation.
    # If `False`: Use TISM to identify the location and mutation.
    location_only: bool
    
    # Number of edits each round.
    # Normally, is N = sequence length.
    # If we use TISM, we can assume we'll be more effective, so we shouldn't
    # need to edit as many locations.
    budget: int
    
    # Fraction of TISM.
    # Use differently in different situations.
    # If `location_only=True`, this is the fraction of locations determined by TISM.
    #   the rest are random.
    # If `location_only=False`, this is the fraction of mutations determined by TISM.
    #   the rest are random location and random mutation.
    fraction_tism: float


def evolve(
    model: mc.ModelClass,
    seqs: list[str],
    max_iter: int,
    batch_size: int = 1,
    positions: list[int] = None,
    verbose: bool = False,
    vocab: list[str] = constants.VOCAB,
    tism_args: Optional[TISMArgs] = None,
) -> tuple[list[str], float, np.ndarray]:
    """
    Sequence design by greedy directed evolution.

    Args:
        model: LightningModel object containing a trained deep learning model
        seqs: a set of DNA sequences as strings or genomic intervals
        max_iter: Number of iterations
        batch_size: Batch size for inference
        return_seqs: "all", "best" or "none".
        positions: Positions to mutate. If None, all positions will be mutated
        verbose: Print status after each iteration.
        tism_args: Optional args to control TISM.

    Returns:
        (best sequences, best score, list of scores)
    """ 
    if positions is None:
        positions = list(range(len(seqs[0])))
    
    # Iteratively perform greedy directed evolution.
    cur_seqs = seqs
    best_score = np.inf
    best_seqs = seqs
    list_of_energies = []
    for i in tqdm.trange(max_iter):
        if verbose:
            print(f"Iteration {i}")
        cur_best_seqs, cur_best_score = single_step(cur_seqs, model, batch_size)
        list_of_energies.append(cur_best_score)
        if verbose:
            # Print the best losses at current iteration
            print(f"Best value at iteration {i}: {cur_best_score:.3f}")
        
        # Check if best sequence is better than the previous best sequence.
        if cur_best_score < best_score:
            best_score = cur_best_score
            best_seqs = cur_best_seqs
        else:
            print(f"Score did not improve on iteration: {i}")

        # Mutate sequences for next iteration.
        if i < max_iter:
            cur_seqs = []
            for seq in best_seqs:
                if tism_args is None:
                    cur_seqs.extend(single_bp_ism(seq, positions, vocab))
                elif tism_args.location_only:
                    cur_positions, must_change_mask = positions_from_tism(
                        seq, model, positions, tism_args)
                    cur_seqs.extend(single_bp_ism(seq, cur_positions, vocab, must_change_mask))
                else:
                    cur_seqs.extend(tism_guided_ism(
                        seq, model, positions, vocab, tism_args))

    return best_seqs, best_score, np.array(list_of_energies)


def single_step(
    cur_seqs: list[str],
    model: mc.ModelClass,
    batch_size: int = 1,
) -> tuple[list[str], float]:
    """A single step in the greedy, directed evolution."""
    # Get predictions.
    preds = get_predictions(
        model=model,
        sequences=cur_seqs,
        batch_size=batch_size,
    )

    # Mark best sequence(s) from current iteration.
    best_score = np.min(preds)
    best_idxs = np.argwhere(preds == best_score)[0]
    best_seqs = [cur_seqs[i] for i in best_idxs]
    
    return best_seqs, best_score

            
def get_predictions(
    model: mc.ModelClass,
    sequences: list[str],
    batch_size: int,
) -> np.ndarray:
    batched_inputs = batchify(sequences, batch_size)
    
    rets = []
    for batched_input in batched_inputs:
        rets.append(model(batched_input))
    
    return np.concatenate(rets, axis=0)
    

def batchify(lst: list, minibatch_size: int):
    """Reshapes a list into batches of a given size."""
    return [lst[i:i + minibatch_size] 
            for i in range(0, len(lst), minibatch_size)]
    
    
def single_bp_ism(
    base_seq: str, 
    positions: list[int], 
    vocab: list[str],
    must_change_mask: Optional[list[bool]] = None,
    ) -> list[str]:
    if must_change_mask is None:
        must_change_mask = [False] * len(positions)
    assert len(positions) == len(must_change_mask)
    
    ret = []
    vocab_set = set(vocab)
    for idx, must_change in zip(positions, must_change_mask):
        if must_change:
            new_char = random.choice(list(vocab_set - set([base_seq[idx]])))
        else:
            new_char = random.choice(list(vocab_set))
        
        new_seq = base_seq[:idx] + new_char + base_seq[idx+1:]
        ret.append(new_seq)
    return ret


PositionType = list[int]
TISMPosMaskType = list[bool]

def positions_from_tism(
    base_seq, 
    model,
    positions, 
    tism_args: TISMArgs,
    ) -> tuple[PositionType, TISMPosMaskType]:
    """Determine positions to mutate based on TISM.
    
    Algo:
    1) Compute TISM
    2) Compute the expected change in energy.
    3) Take the top N
    4) Pick the remainder at random from the remaining locations.
    """
    assert tism_args.location_only, tism_args
    assert len(positions) >= tism_args.budget
    
    _, tism_list = model.tism(base_seq)
    expected_energy_change_and_pos = [
        (np.mean(list(d.values())), i) for i, d in enumerate(tism_list)]
    expected_energy_change_and_pos = sorted(expected_energy_change_and_pos)
    
    # Select positions to edit based on the above.
    num_tism_positions = int(tism_args.budget * tism_args.fraction_tism)
    num_random_positions = tism_args.budget - num_tism_positions
    
    # Fill TISM positions until we reach the quota, as long as positions are in the preapproved list.
    tism_positions = []
    for _, candidate_pos in expected_energy_change_and_pos:
        if candidate_pos in positions:
            tism_positions.append(candidate_pos)
        if len(tism_positions) >= num_tism_positions:
            break

    # Sample should be without replacement.
    remaining_positions = list(set(positions) - set(tism_positions))
    random_positions = random.sample(remaining_positions, k=num_random_positions)
    
    positions = tism_positions + random_positions
    tism_mask = [True] * num_tism_positions + [False] * num_random_positions
    
    assert len(positions) == len(tism_mask)
    
    return positions, tism_mask

def tism_guided_ism(
    base_seq: str, 
    model,
    positions: list[int], 
    vocab: list[str],
    tism_args: TISMArgs,
    ) -> list[str]:
    """Select positions to mutate based on TISM.
    
    General flow:
    1) Compute TISM
    2) Pick the top N mutations accoring to TISM
    3) Pick the remainder at random
    """
    assert tism_args.location_only is False, tism_args
    assert len(positions) >= tism_args.budget
    
    _, tism_list = model.tism(base_seq)
    
    energy_change_pos_mutation = [(v, i, k) for i, d in enumerate(tism_list) for k, v in d.items()]
    energy_change_pos_mutation = sorted(energy_change_pos_mutation)
    
    # Select positions to edit based on the above.
    num_tism_positions = int(tism_args.budget * tism_args.fraction_tism)
    num_random_positions = tism_args.budget - num_tism_positions
    
    # Fill TISM positions until we reach the quota, as long as positions are in the preapproved list.
    mutations, tism_positions = [], []
    for _, candidate_pos, candidate_mutation in energy_change_pos_mutation:
        if candidate_pos in positions:
            mutations.append((candidate_pos, candidate_mutation))
            tism_positions.append(candidate_pos)
        if len(mutations) >= num_tism_positions:
            break
        
    # Sample should be without replacement.
    remaining_positions = list(set(positions) - set(tism_positions))
    random_positions = random.sample(remaining_positions, k=num_random_positions)
    for pos in random_positions:
        mutations.append((pos, random.choice(vocab)))
        
    assert len(mutations) == tism_args.budget
    
    ret = []
    for idx, new_char in mutations:
        new_seq = base_seq[:idx] + new_char + base_seq[idx+1:]
        ret.append(new_seq)
    return ret