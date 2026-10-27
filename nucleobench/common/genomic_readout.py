"""Validated sequence inputs and overlap-weighted genomic readouts.

Shared by the long-context genomic models (Borzoi, AlphaGenome), which score a
gene by averaging a predicted coverage profile over a set of exons.
"""

import hashlib
from pathlib import Path

import torch

from nucleobench.common import constants, string_utils

# `N` is encoded as an all-zero column. We map it to `A` before one-hot encoding
# and zero the column afterwards, so that we can reuse `dna2tensor_batch`.
_AMBIGUOUS = "N"
_N_TO_A = str.maketrans(_AMBIGUOUS, "A")
_ALLOWED = set(constants.VOCAB) | {_AMBIGUOUS}


def one_hot(sequences: list[str], length: int, device: str = "cpu") -> torch.Tensor:
    """One-hot encode DNA strings of exactly `length` into (batch, 4, length).

    Lowercase sequence is accepted. `N` becomes an all-zero column.
    """
    if not isinstance(sequences, (list, tuple)) or not sequences:
        raise ValueError("Expected a nonempty list of DNA strings")
    normalized = []
    for sequence in sequences:
        if not isinstance(sequence, str) or len(sequence) != length:
            raise ValueError(f"Each DNA string must have length {length}")
        sequence = sequence.upper()
        bad = set(sequence) - _ALLOWED
        if bad:
            raise ValueError(f"Invalid DNA base {sorted(bad)[0]!r} in sequence")
        normalized.append(sequence)

    tensor = string_utils.dna2tensor_batch(
        [sequence.translate(_N_TO_A) for sequence in normalized]
    )
    for i, sequence in enumerate(normalized):
        if _AMBIGUOUS in sequence:
            ambiguous = [j for j, base in enumerate(sequence) if base == _AMBIGUOUS]
            tensor[i, :, ambiguous] = 0
    return tensor.to(device)


def validate_tensor(x: torch.Tensor, length: int) -> None:
    if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1:] != (4, length):
        raise ValueError(f"Expected a tensor shaped (batch, 4, {length})")
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("DNA tensor must contain finite floating-point values")


def union_intervals(intervals, lower: int, upper: int) -> list[tuple[int, int]]:
    """Merge overlapping half-open intervals, all of which must lie in bounds."""
    if not intervals:
        raise ValueError("At least one exon interval is required")
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not (
            isinstance(start, int)
            and isinstance(end, int)
            and lower <= start < end <= upper
        ):
            raise ValueError(f"Exon [{start}, {end}) is outside [{lower}, {upper})")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def coverage_weights(
    n_bins: int,
    intervals,
    *,
    origin: int,
    resolution: int,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Base pairs each of `n_bins` bins contributes to the union of `intervals`.

    Depends only on the exons and the output grid, so a model that scores many
    sequences against the same gene builds this once and reuses it with
    `weighted_mean` instead of rebuilding it on every forward pass.
    """
    merged = union_intervals(intervals, origin, origin + resolution * n_bins)
    weights = torch.zeros(n_bins, dtype=dtype, device=device)
    bin_starts = origin + torch.arange(n_bins, device=device) * resolution
    bin_ends = bin_starts + resolution
    for start, end in merged:
        weights += (
            torch.minimum(bin_ends, torch.as_tensor(end, device=device))
            - torch.maximum(bin_starts, torch.as_tensor(start, device=device))
        ).clamp_min(0)
    return weights


def weighted_mean(profile: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Mean of a (batch, bins) profile under precomputed per-bin weights."""
    if profile.ndim != 2 or profile.shape[1] != weights.shape[0]:
        raise ValueError(f"Expected a (batch, {weights.shape[0]}) profile")
    # A no-op when the weights were built for this profile's device and dtype.
    weights = weights.to(device=profile.device, dtype=profile.dtype)
    return (profile * weights).sum(dim=1) / weights.sum()


def coverage_mean(
    profile: torch.Tensor, intervals, *, origin: int, resolution: int
) -> torch.Tensor:
    """Mean of a (batch, bins) profile over the union of half-open intervals.

    Bins are weighted by how many base pairs they contribute to the union, so
    that exon boundaries falling inside a bin are handled correctly.
    """
    if profile.ndim != 2:
        raise ValueError("Expected (batch, bins) profile")
    return weighted_mean(
        profile,
        coverage_weights(
            profile.shape[1],
            intervals,
            origin=origin,
            resolution=resolution,
            dtype=profile.dtype,
            device=profile.device,
        ),
    )


def file_sha256(path: str | Path) -> str:
    """SHA256 of a file, read in chunks so that large checkpoints fit in memory."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_exons(value: str) -> list[tuple[int, int]]:
    """Parse a `start:end,start:end` command-line value into intervals."""
    intervals = []
    for item in value.split(","):
        start, end = item.split(":")
        intervals.append((int(start), int(end)))
    return intervals
