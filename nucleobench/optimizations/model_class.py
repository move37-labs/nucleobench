"""Parent class for models."""

from typing import Any, Callable

import numpy as np
import torch

from nucleobench.common import attribution_lib_torch as att_lib
from nucleobench.common import constants, string_utils

SequenceType = str


class ModelClass:
    @staticmethod
    def init_parser():
        raise ValueError("Not implemented.")

    @staticmethod
    def debug_init_args() -> dict[str, Any]:
        raise ValueError("Not implemented.")

    def __init__(self, model_fn: Callable, start_sequence: SequenceType):
        raise NotImplementedError("Not implemented.")

    def __call__(self, x: list[str], return_debug_info: bool = False) -> np.ndarray | tuple[np.ndarray, Any]:
        """Takes in a list of strings, returns a scalar value per string."""
        raise NotImplementedError("Not implemented.")


class TISMModelClass(ModelClass):
    """Model that supports TISM."""

    vocab: list[str]

    def inference_on_tensor(
        self, x: torch.Tensor, return_debug_info: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        raise NotImplementedError("Not implemented.")

    def tism(
        self, x: str, idxs: list[int] | None = None
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """Runs Taylor in-silico mutagenesis on inputs.

        Deprecated. Use `get_tism` instead.
        """
        try:
            cur_vocab = self.vocab
        except AttributeError:
            cur_vocab = constants.VOCAB

        input_tensor = string_utils.dna2tensor(x, vocab_list=cur_vocab)
        sg_tensor = att_lib.grad_torch(
            input_tensor=torch.unsqueeze(input_tensor, dim=0),
            model=self.inference_on_tensor,  # type: ignore[arg-type]
            idxs=idxs,
        )
        sg = att_lib.grad_tensor_to_dict(
            torch.squeeze(sg_tensor, dim=0), vocab=cur_vocab
        )
        x_effective = x if idxs is None else "".join([x[idx] for idx in idxs])
        sg = att_lib.grad_to_tism(sg, x_effective)  # type: ignore[assignment]
        y = self.inference_on_tensor(torch.unsqueeze(input_tensor, dim=0))
        return y, sg  # type: ignore[return-value]

    def str2tensor(self, x: str) -> torch.Tensor:
        """Convert a string to a tensor.

        It must be done in a predictable way, so that we can efficiently manipulate
        the Tensor, then consistently converted back.

        Child classes can override this method if needed.
        """
        assert hasattr(self, "vocab"), "Vocab not set."
        return string_utils.dna2tensor(x, vocab_list=self.vocab)

    def tensor2int(self, x: str) -> torch.Tensor:
        """Convert a string sequence to an integer-encoded tensor.

        It must be done in a predictable way, so that we can efficiently manipulate
        the Tensor, then consistently converted back.

        Child classes can override this method if needed.
        """
        assert hasattr(self, "vocab"), "Vocab not set."
        return string_utils.dna2tensor_integer(x, vocab_list=self.vocab)

    def tism_torch(self, x: str, idxs: list[int] | None = None) -> torch.Tensor:
        input_tensor = self.str2tensor(x)
        sg_tensor = att_lib.grad_torch(
            input_tensor=torch.unsqueeze(input_tensor, dim=0),
            model=self.inference_on_tensor,  # type: ignore[arg-type]
            idxs=idxs,
        )
        # Determine the effective sequence (full or subset by idxs)
        if idxs is None:
            x_effective = x
        else:
            x_effective = "".join([x[idx] for idx in idxs])
        base_seq_idx = self.tensor2int(x_effective)
        tism_tensor = att_lib.grad_torch_to_tism_torch(
            torch.squeeze(sg_tensor, dim=0), base_seq_idx
        )
        return tism_tensor

    def get_tism(
        self, sequence: str, idxs: list[int] | None = None
    ) -> tuple[list[tuple[Any, Any]], np.ndarray]:
        assert hasattr(self, "vocab_to_idx"), (
            f'{self.__class__.__name__}: missing "vocab_to_idx".'
        )
        assert hasattr(self, "vocab_array"), (
            f'{self.__class__.__name__}: missing "vocab_array".'
        )
        tism_tensor = self.tism_torch(sequence, idxs)
        vocab_size, tism_seq_len = tism_tensor.shape

        # Determine positions to mutate
        if idxs is None:
            positions_to_mutate = np.arange(len(sequence), dtype=np.int32)
        else:
            positions_to_mutate = np.array(idxs, dtype=np.int32)

        assert len(positions_to_mutate) == tism_seq_len, (
            f"Length mismatch: positions_to_mutate={len(positions_to_mutate)}, tism_seq_len={tism_seq_len}"
        )

        # VECTORIZED OPTIMIZATION: Convert to numpy once and use vectorized operations
        # Check device to avoid unnecessary CPU transfer
        if tism_tensor.device.type != "cpu":
            tism_np = tism_tensor.cpu().numpy()
        else:
            tism_np = tism_tensor.numpy()

        # Build base sequence indices for each position (using integer indices, not strings)
        base_seq_chars = np.array([sequence[pos] for pos in positions_to_mutate])
        base_seq_indices = np.array(
            [self.vocab_to_idx[char] for char in base_seq_chars]
        )

        # Create all possible (position, nucleotide) pairs using vectorized operations
        # positions_array: [pos0, pos0, pos0, pos0, pos1, pos1, pos1, pos1, ...]
        positions_array = np.repeat(positions_to_mutate, vocab_size)

        # vocab_repeated: [A, C, G, T, A, C, G, T, ...] (repeated for each position)
        vocab_repeated = np.tile(self.vocab_array, tism_seq_len)

        # Create indices for tism_np lookup: (vocab_idx, pos_idx) for each pair
        vocab_indices = np.tile(
            np.arange(vocab_size), tism_seq_len
        )  # [0, 1, 2, 3, 0, 1, 2, 3, ...]
        pos_indices = np.repeat(
            np.arange(tism_seq_len), vocab_size
        )  # [0, 0, 0, 0, 1, 1, 1, 1, ...]

        # Get TISM values for all pairs (vectorized extraction)
        tism_values = tism_np[vocab_indices, pos_indices]
        assert tism_values.shape == (len(positions_array),)

        # Create mask: exclude pairs where nucleotide == base_char (using integer comparison, not strings)
        base_seq_indices_expanded = np.repeat(base_seq_indices, vocab_size)
        valid_mask = vocab_indices != base_seq_indices_expanded

        # Extract valid pairs using boolean indexing (fully vectorized)
        valid_positions = positions_array[valid_mask]
        valid_vocab = vocab_repeated[valid_mask]
        valid_logits = tism_values[valid_mask]

        # Convert to required format - optimized: use tolist() for faster conversion
        # Converting numpy arrays to Python lists first is faster than element-wise conversion
        pos_and_chars_to_mutate = list(
            zip(valid_positions.tolist(), valid_vocab.tolist())
        )
        logits = valid_logits.astype(np.float32)

        return (pos_and_chars_to_mutate, logits)


class PyTorchDifferentiableModel(ModelClass):
    """Model that can produce differentiable, PyTorch tensors."""

    def inference_on_tensor(
        self, x: torch.Tensor, return_debug_info: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        raise NotImplementedError("Not implemented.")
