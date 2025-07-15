"""Utils for manipulating stringsl."""

import os
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import subprocess

from nucleobench.common import constants


def dna2tensor(sequence_str: str, vocab_list: list[str] = constants.VOCAB) -> torch.Tensor:
    """
    Convert a DNA sequence to a one-hot encoded tensor.

    Args:
        sequence_str (str): DNA sequence string.
        vocab_list (list): List of DNA nucleotide characters.

    Returns:
        torch.Tensor: One-hot encoded tensor representation of the sequence.
    """
    # Dictionary lookup is faster. Can matter in performance, since this method can be bottleneck.
    vocab_map = {nt: i for i, nt in enumerate(vocab_list)}
    
    # 1. Convert the string sequence to a tensor of integer indices.
    # The dictionary lookup is O(1) and faster than list.index().
    int_tensor = torch.tensor([vocab_map[c] for c in sequence_str], dtype=torch.long)

    # 2. Use F.one_hot for efficient conversion.
    # It creates a tensor of shape (sequence_length, num_classes).
    one_hot_tensor = F.one_hot(int_tensor, num_classes=len(vocab_list))

    # 3. Transpose to (num_classes, sequence_length) and convert to float
    # to match the original function's output format.
    return one_hot_tensor.T.float()


def dna2tensor_batch(sequence_strs: list[str], vocab_list: list[str] = constants.VOCAB) -> torch.Tensor:
    """
    Efficiently convert a batch of DNA sequences to a one-hot encoded tensor.
    Assumes all sequences in the batch are the same length.

    Args:
        sequence_strs (list[str]): A list of DNA sequence strings.

    Returns:
        torch.Tensor: One-hot encoded tensor of shape
                      (batch_size, vocab_size, sequence_length).
    """
    # Dictionary lookup is faster. Can matter in performance, since this method can be bottleneck.
    vocab_map = {nt: i for i, nt in enumerate(vocab_list)}
    
    # 1. Convert the list of strings to a 2D tensor of integer indices.
    # This is done in a single tensor creation call.
    int_tensor = torch.tensor([[vocab_map[c] for c in seq] for seq in sequence_strs],
                              dtype=torch.long)
    # The resulting tensor has shape: (batch_size, sequence_length)

    # 2. Apply one-hot encoding to the entire batch tensor at once.
    # F.one_hot adds a new dimension at the end.
    # Output shape: (batch_size, sequence_length, vocab_size)
    one_hot_tensor = F.one_hot(int_tensor, num_classes=len(vocab_list))

    # 3. Permute the dimensions to match the desired output shape and set type.
    # We swap the last two dimensions to get (batch_size, vocab_size, sequence_length)
    return one_hot_tensor.permute(0, 2, 1).float()


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
