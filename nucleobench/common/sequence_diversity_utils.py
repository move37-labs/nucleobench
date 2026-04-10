"""Utilities for computing sequence diversity."""

import numpy as np


def pairwise_hamming_distance(seqs: list[str]) -> float:
    """
    Compute the average pairwise Hamming distance between sequences using numpy vectorization.
    
    This is much faster than the nested loop version, especially for large groups.
    
    NOTE: These values are not normalized by sequence length. That must happen later.
    """
    n = len(seqs)
    if n < 2:
        raise ValueError('Must have at least 2 proposals to compute pairwise Hamming distance.')
    
    # 2. Fast Convert to Bytes (uint8)
    # This avoids creating Python list-of-lists overhead
    # view('S1').reshape((n, -1)) creates a (N, L) array of ASCII bytes
    try:
        seq_array = np.array(seqs, dtype='S').view('S1').reshape((n, -1))
    except ValueError:
        raise ValueError("All sequences must be the same length for vectorized Hamming distance.")
    
    # 3. Vectorized Broadcasting
    # (N, 1, L) != (1, N, L) -> (N, N, L) boolean mask
    # Sum over Length axis -> (N, N) distance matrix
    hamming_matrix = (seq_array[:, None] != seq_array).sum(axis=2)
    
    # 4. Extract Upper Triangle
    # We use the indices to grab only the unique pairs (i < j)
    # This avoids double counting and the zero-diagonal
    upper_triangle_values = hamming_matrix[np.triu_indices(n, k=1)]
    
    hamming_metric = float(upper_triangle_values.mean())
    
    # Optional normalization by length if you want a percentage
    # seq_len = seq_array.shape[1]
    # hamming_metric /= seq_len 
    
    print(f'Average Hamming Dist: {hamming_metric:.3f} (N={n})')
    return hamming_metric