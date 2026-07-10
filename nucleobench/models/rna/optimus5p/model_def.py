import argparse

import numpy as np
import torch

from nucleobench.common import string_utils
from nucleobench.optimizations import model_class as mc

from . import load_model


class Optimus5P(mc.PyTorchDifferentiableModel, mc.TISMModelClass):
    """Optimus 5-Prime model for mean ribosome load prediction."""

    @staticmethod
    def init_parser():
        """
        Add arguments to an argparse ArgumentParser.

        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.

        Returns:
            argparse.ArgumentParser: Argument parser with added arguments.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--window_size", type=int, default=50)
        parser.add_argument("--stride", type=int, default=5)
        parser.add_argument("--override_weights_local_path", type=str, default=None)
        return parser

    @staticmethod
    def debug_init_args():
        return {
            "window_size": 50,
            "stride": 5,
        }

    def __init__(
        self,
        window_size: int = 50,
        stride: int = 5,
        override_weights_local_path: str | None = None,
        override_model: torch.nn.Module | None = None,
    ):
        self.window_size = window_size
        self.stride = stride
        self.vocab = ["A", "C", "G", "T"]
        self.vocab_to_idx = {nt: i for i, nt in enumerate(self.vocab)}
        self.vocab_array = np.array(self.vocab)
        self.has_cuda = torch.cuda.is_available()

        if override_model:
            self.model = override_model
        else:
            self.model = load_model.load_model(
                weights_path=override_weights_local_path,
                has_cuda=self.has_cuda,
            )
        self.model.eval()

    def inference_on_tensor(self, x: torch.Tensor, return_debug_info: bool = False) -> torch.Tensor:
        """Run inference on a one-hot encoded tensor.

        Args:
            x: One-hot encoded tensor of shape (batch_size, 4, seq_len)

        Returns:
            Tensor of shape (batch_size,) with negative predictions (to minimize)
        """
        assert x.ndim == 3 and x.shape[1] == 4, (
            f"Expected one-hot tensor with shape (batch, 4, seq_len), got {x.shape}"
        )

        batch_size, _, seq_len = x.shape

        if seq_len == self.window_size:
            ret = self.model(x).squeeze(1)
        elif seq_len < self.window_size:
            # Pad with zeros to self.window_size
            padding = self.window_size - seq_len
            padded_x = torch.nn.functional.pad(x, (0, padding))
            ret = self.model(padded_x).squeeze(1)
        else:
            # Sliding window on tensor
            windows = []
            max_start_index = seq_len - self.window_size
            for i in range(0, max_start_index + 1, self.stride):
                windows.append(x[:, :, i : i + self.window_size])

            last_window = x[:, :, -self.window_size :]
            if max_start_index % self.stride != 0:
                windows.append(last_window)
            elif not windows:
                windows.append(last_window)

            stacked_windows = torch.stack(
                windows, dim=0
            )  # (num_windows, batch_size, 4, window_size)
            num_windows = stacked_windows.shape[0]

            flat_windows = stacked_windows.view(
                num_windows * batch_size, 4, self.window_size
            )
            flat_scores = self.model(flat_windows).squeeze(
                1
            )  # (num_windows * batch_size,)

            scores_per_window = flat_scores.view(num_windows, batch_size)
            ret = scores_per_window.mean(dim=0)  # (batch_size,)

        # Multiply by -1 so "better" sequences are lower, according to convention.
        return -1 * ret

    def inference_on_strings(self, x: list[str]) -> np.ndarray:
        # Convert RNA/DNA strings to one-hot tensors
        seqs_dna = [seq.replace("U", "T") for seq in x]
        batch_onehot = string_utils.dna2tensor_batch(seqs_dna, vocab_list=self.vocab)

        if self.has_cuda:
            batch_onehot = batch_onehot.cuda()

        ret = self.inference_on_tensor(batch_onehot)
        return ret.detach().clone().cpu().numpy()

    def __call__(self, x: list[str], return_debug_info: bool = False) -> np.ndarray:
        if isinstance(x, str):
            raise ValueError(f"Input needs to be list of strings, not just string: {x}")
        return self.inference_on_strings(x)
