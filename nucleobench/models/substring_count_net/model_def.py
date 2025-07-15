"""Model for counting substrings."""

import argparse
import torch
import torch.nn.functional as F
from typing import Optional

from nucleobench.common import argparse_lib
from nucleobench.common import constants
from nucleobench.common import attribution_lib_torch as att_lib
from nucleobench.common import string_utils

from nucleobench.optimizations import model_class as mc

class CountSubstringModel(torch.nn.Module, mc.PyTorchDifferentiableModel, mc.TISMModelClass):
    """Count number of substrings, using convs."""
    def __init__(self,
                 substring: str,
                 tism_times: int = 3,
                 tism_stdev: float = 0.25,
                 flip_sign: bool = False,
                 vocab: list[str] = constants.VOCAB):
        super().__init__()
        self.substring = substring
        self.tism_times = tism_times
        self.tism_stdev = tism_stdev
        self.flip_sign = flip_sign
        self.vocab = vocab

        self.substr_tensor = string_utils.dna2tensor(
            substring, vocab_list=self.vocab)
        self.substr_tensor = torch.unsqueeze(self.substr_tensor, dim=0)
        self.substr_tensor.requires_grad = False

    @staticmethod
    def init_parser():
        """
        Add energy-specific arguments to an argparse ArgumentParser.

        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.

        Returns:
            argparse.ArgumentParser: Argument parser with added energy-specific arguments.

        """
        parser = argparse.ArgumentParser()
        group = parser.add_argument_group("Substring count init args")
        group.add_argument("--substring", type=str, required=True)
        group.add_argument("--tism_times", type=int, default=3)
        group.add_argument("--tism_stdev", type=float, default=0.25)
        group.add_argument("--flip_sign", type=argparse_lib.str_to_bool, default=False)

        return parser

    @staticmethod
    def debug_init_args():
        return {
            'substring': 'AG',
            'flip_sign': True,
        }


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        assert x.shape[1] == 4, x.shape
        out_tensor = F.conv1d(x, self.substr_tensor)
        out_tensor = torch.squeeze(out_tensor, 1)
        # We square it so it's nonlinear. That is, getting all 3 in one window should be
        # better than getting 2 in one window and 1 in another.
        out_tensor = torch.square(out_tensor)
        out_tensor = torch.sum(out_tensor, dim=1)

        if self.flip_sign:
            out_tensor *= -1

        return out_tensor

    def inference_on_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def tism(self, x: str, idxs: Optional[list[int]] = None) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        input_tensor = string_utils.dna2tensor(x, vocab_list=self.vocab)
        sg_tensor = att_lib.smoothgrad_torch(
            input_tensor=input_tensor,
            model=self.inference_on_tensor,
            noise_stdev=self.tism_stdev,
            times=self.tism_times,
            idxs=idxs,
        )
        sg = att_lib.smoothgrad_tensor_to_dict(sg_tensor, vocab=self.vocab)
        x_effective = x if idxs is None else [x[idx] for idx in idxs]
        sg = att_lib.smoothgrad_to_tism(sg, x_effective)
        y = self.inference_on_tensor(torch.unsqueeze(input_tensor, dim=0))
        return y, sg

    def __call__(self, seqs: list[str], return_debug_info: bool = False):
        torch_seq = string_utils.dna2tensor_batch(seqs)
        result = self.inference_on_tensor(torch_seq)
        assert result.ndim == 1, result.shape
        if return_debug_info:
            return [float(x) for x in result], {}
        else:
            return [float(x) for x in result]
