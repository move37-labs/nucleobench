"""Utils for manipulating stringsl."""

import os
import numpy as np
import torch
from typing import Union
import subprocess

from nucleobench.common import constants


def dna2tensor(sequence_str: str, vocab_list=constants.VOCAB) -> torch.Tensor:
    """
    Convert a DNA sequence to a one-hot encoded tensor.

    Args:
        sequence_str (str): DNA sequence string.
        vocab_list (list): List of DNA nucleotide characters.

    Returns:
        torch.Tensor: One-hot encoded tensor representation of the sequence.
    """
    seq_tensor = np.zeros((len(vocab_list), len(sequence_str)))
    for letterIdx, letter in enumerate(sequence_str):
        seq_tensor[vocab_list.index(letter), letterIdx] = 1
    seq_tensor = torch.Tensor(seq_tensor)
    return seq_tensor


def dna2tensor_batch(
    sequence_strs: list[str], vocab_list=constants.VOCAB
) -> torch.Tensor:
    """
    Convert a DNA sequence to a one-hot encoded tensor.

    Args:
        sequence_str (str): DNA sequence string.
        vocab_list (list): List of DNA nucleotide characters.

    Returns:
        torch.Tensor: One-hot encoded tensor representation of the sequence.
    """
    seq_tensors = [dna2tensor(x, vocab_list) for x in sequence_strs]
    return torch.stack(seq_tensors)


def tensor2dna(
    tensor: Union[torch.Tensor, np.ndarray], vocab_list=constants.VOCAB
) -> str:
    """
    Convert a one-hot encoded tensor to a DNA sequence.

    Args:
        tensor: One-hot encoded Tensor or array.
        vocab_list (list): List of DNA nucleotide characters.

    Returns:
        torch.Tensor: One-hot encoded tensor representation of the sequence.
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy()

    if tensor.ndim != 2:
        raise ValueError(f"Expected dim 2: {tensor.ndim}")
    if tensor.shape[0] != len(vocab_list):
        raise ValueError(f"Expected vocab dim to be {len(vocab_list)}): {tensor.shape}")
    if not np.all(np.isin(tensor, [0, 1])):
        raise ValueError(f"Expected values to be 0 or 1: {tensor}")
    col_sum = np.sum(tensor, axis=0)
    if not np.all(col_sum == np.ones_like(col_sum)):
        raise ValueError(f"Not onehot: {tensor}")

    # Convert one-hot tensors to string sequence.
    seq = []
    for idx in range(tensor.shape[1]):
        cur_onehot = tensor[:, idx]
        cur_char = vocab_list[np.nonzero(cur_onehot)[0][0]]
        seq.append(cur_char)
    return "".join(seq)


def tensor2dna_batch(
    tensor: Union[torch.Tensor, np.ndarray], vocab_list=constants.VOCAB
) -> list[str]:
    """
    Convert a one-hot encoded tensor to a DNA sequence.

    Args:
        tensor: One-hot encoded Tensor or array.
        vocab_list (list): List of DNA nucleotide characters.

    Returns:
        torch.Tensor: One-hot encoded tensor representation of the sequence.
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy()
    if tensor.ndim != 3:
        raise ValueError(f"Expected dim 3: {tensor.ndim}")

    return [tensor2dna(t, vocab_list) for t in tensor]


def str2np(sequence_str: str, vocab_list=constants.VOCAB) -> np.ndarray:
    """
    Convert a DNA sequence string to a numpy array.

    Args:
        sequence_str (str): DNA sequence string. Each character is a nucleotide (e.g. A, C, T, G).
        vocab_list (list): List of DNA nucleotide characters.

    Returns:
        np.ndarray: DNA sequence array. Each array element is a nucleotide index (e.g. 0, 1, 2, 3).
    """
    return np.array([vocab_list.index(letter) for letter in sequence_str])


def np2str(sequence_np: np.ndarray, vocab_list=constants.VOCAB) -> str:
    """
    Convert a DNA sequence array to a string.

    Args:
        sequence_np (np.ndarray): DNA sequence array. Each array element is a nucleotide index (e.g. 0, 1, 2, 3).
        vocab_list (list): List of DNA nucleotide characters.

    Returns:
        str: DNA sequence string. Each character is a nucleotide (e.g. A, C, T, G).
    """
    return "".join([vocab_list[letter] for letter in sequence_np])


SeqOrSeqsType = Union[str, list[str]]


def load_sequences(
    artifact_path_or_seq: str, download_path: str = "./"
) -> tuple[SeqOrSeqsType, str]:
    """Load start sequences from file or gcs. Can be a single sequence or a list of sequences.

    Use the same download style as other places in this file.

    Input can either be:
        - a local file path
        - a gcs file path
        - a single sequence
        - a comma delimited list of sequences

    Returns:
        (sequence or sequences, a comment on where the sequence came from.)
    """
    ret_comment = None
    if not set(artifact_path_or_seq).issubset(set(constants.VOCAB + [","])):
        # Assume it's a path.
        artifact_path = artifact_path_or_seq
        ret_comment = artifact_path
        if artifact_path.startswith("gs://"):
            # TODO(joelshor): Read using google.storage, not subprocess.
            subprocess.call(["gsutil", "cp", artifact_path, download_path])
            artifact_path = os.path.join(download_path, os.path.basename(artifact_path))
        with open(artifact_path, "r") as f:
            seq_or_seqs = f.read().strip()
    else:
        seq_or_seqs = artifact_path_or_seq
        if "," in seq_or_seqs:
            ret_comment = "From command line, comma delimited."
        else:
            ret_comment = "From command line."

    # Determine whether it's a single sequence or a list of sequences.
    if "," in seq_or_seqs:
        seq_or_seqs = seq_or_seqs.split(",")

    assert ret_comment is not None
    return seq_or_seqs, ret_comment
