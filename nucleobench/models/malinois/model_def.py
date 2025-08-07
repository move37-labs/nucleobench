"""Public interface for Malinois model.

To test on real data:
```zsh
python -m nucleobench.models.malinois.model_def
```
"""

from typing import Optional, Union

import argparse
import gc
import numpy as np
import torch

from nucleobench.common import constants
from nucleobench.common import string_utils
from nucleobench.common import attribution_lib_torch as att_lib

from nucleobench.optimizations import model_class as mc
from nucleobench.models.malinois import load_model


class Malinois(mc.PyTorchDifferentiableModel, mc.TISMModelClass):
    """Malinois model using MinGap energy."""

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
        group = parser.add_argument_group("Malinois init args")
        group.add_argument(
            "--model_artifact", type=str,
            default="gs://tewhey-public-data/CODA_resources/malinois_artifacts__20211113_021200__287348.tar.gz",
        )
        group.add_argument("--target_feature", type=int)
        group.add_argument("--bending_factor", type=float)
        group.add_argument("--a_min", type=float, default=-2.0)
        group.add_argument("--a_max", type=float, default=6.0)
        group.add_argument("--target_alpha", type=float, default=1.0)
        group.add_argument("--flank_length", type=int, default=200)
        # Smoothgrad args.
        

        return parser
    
    @staticmethod
    def debug_init_args():
        return {
            'target_feature': 0,
            'bending_factor': 0.0,
            'check_input_shape': True,
        }

    def __init__(
        self,
        target_feature: int,
        bending_factor: float,
        model_artifact: str = "gs://tewhey-public-data/CODA_resources/malinois_artifacts__20211113_021200__287348.tar.gz",
        a_min: Optional[float] = -2.0,
        a_max: Optional[float] = 6.0,
        target_alpha: float = 1.0,
        flank_length: int = 200,
        vocab: list[str] = constants.VOCAB,
        override_model: Optional[torch.nn.Module] = None,
        check_input_shape: bool = False,
    ):
        self.has_cuda = torch.cuda.is_available()
        if override_model:
            self.model = override_model
        else:
            self.model = load_model.load_model(model_artifact, has_cuda=self.has_cuda)

        self._model_artifact = model_artifact
        self.target_feature = target_feature
        self.bending_factor = bending_factor
        self.a_min = a_min
        self.a_max = a_max
        self.target_alpha = target_alpha
        self.check_input_shape = check_input_shape

        self.flank_length = flank_length
        if self.flank_length > 0:
            self.left_flank = constants.MPRA_UPSTREAM[-self.flank_length:]
            self.right_flank = constants.MPRA_DOWNSTREAM[:self.flank_length]
        else:
            self.left_flank = None
            self.right_flank = None
        # Consistent vocab is important for interpreting smoothgrad.
        self.vocab = vocab

        if self.left_flank:
            self.left_flank_tensor = string_utils.dna2tensor(
                self.left_flank, vocab_list=self.vocab
            )
            self.left_flank_tensor.requires_grad = False
            assert list(self.left_flank_tensor.shape) == [4, self.flank_length]
        else:
            self.left_flank_tensor = None

        if self.right_flank:
            self.right_flank_tensor = string_utils.dna2tensor(
                self.right_flank, vocab_list=self.vocab
            )
            self.right_flank_tensor.requires_grad = False
            assert list(self.right_flank_tensor.shape) == [4, self.flank_length]
        else:
            self.right_flank_tensor = None

    def inference_on_tensor(
        self, 
        x: torch.Tensor,
        return_debug_info: bool = False,
        ) -> torch.Tensor:
        """Run inference on a one-hot tensor."""
        assert x.ndim == 3  # Batched.
        x_flanked = self.add_flanks_tensor(x)
        if self.has_cuda:
            x_flanked = x_flanked.cuda()
        if self.check_input_shape and x_flanked.shape[2] != 600:
            raise ValueError(f'Malinois input wrong size: {x.shape} {x_flanked.shape}')
        m_out = self.model(x_flanked)
        ret_energy = energy_calc_from_output_tensor(
            model_output_tensor=m_out,
            target_feature=self.target_feature,
            bending_factor=self.bending_factor,
            a_min=self.a_min,
            a_max=self.a_max,
            target_alpha=self.target_alpha,
        )
        
        if return_debug_info:
            return ret_energy, {'malinois_output': m_out}
        else:
            return ret_energy


    def add_flanks_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Add Tensor flanks, with the right backprop properties."""
        bdim = x.shape[0]
        if self.left_flank_tensor is not None:
            left_stack = torch.stack([self.left_flank_tensor] * bdim)
            left_stack.requires_grad = False
        else:
            left_stack = None
        if self.right_flank_tensor is not None:
            right_stack = torch.stack([self.right_flank_tensor] * bdim)
            right_stack.requires_grad = False
        else:
            right_stack = None

        if left_stack is not None and right_stack is not None:
            return torch.concat([left_stack, x, right_stack], dim=2)
        elif left_stack is not None:
            return torch.concat([left_stack, x], dim=2)
        elif right_stack is not None:
            return torch.concat([x, right_stack], dim=2)
        else:
            return x

    def inference_on_strings(
        self, 
        x: list[str], 
        return_debug_info: bool = False,
        ) -> np.ndarray:
        tensor = string_utils.dna2tensor_batch(x, vocab_list=self.vocab)
        ret = self.inference_on_tensor(tensor, return_debug_info)
        if return_debug_info:
            return ret[0].detach().clone().numpy(), ret[1]
        else:
            return ret.detach().clone().numpy()

    def add_flank_string(self, x: str) -> str:
        """Unused, in favor of flanking in tensors."""
        if self.left_flank and self.right_flank:
            return self.left_flank + x + self.right_flank
        elif self.left_flank:
            return self.left_flank + x
        elif self.right_flank:
            return x + self.right_flank
        else:
            return x

    def add_flank_strings(self, xs: list[str]) -> list[str]:
        """Unused, in favor of flanking in tensors."""
        return [self.add_flank_string(x) for x in xs]

    def __call__(
        self, 
        x: list[str], 
        return_debug_info: bool = False, 
        ) -> np.ndarray:
        if isinstance(x, str):
            raise ValueError(f'Malinois input needs to be list of strings, not just string: {x}')
        return self.inference_on_strings(x, return_debug_info)


def _bend(x: torch.Tensor, bending_factor: float) -> torch.Tensor:
    """A 'bending' function, taken from Boda2 paper."""
    return x - bending_factor * (torch.exp(-x) - 1)


def energy_calc_from_output_tensor(
    model_output_tensor: torch.Tensor,
    target_feature: int,
    bending_factor: float,
    a_min: Optional[float] = -2.0,
    a_max: Optional[float] = 6.0,
    target_alpha: float = 1.0,
) -> torch.Tensor:
    """
    Calculate the energy of input sequences based on the maximum model outputs.

    Args:
        model_output_tensor (torch.Tensor): Output tensor, 2D [batch x cell type]
        target_feature: Int representing the desired cell-type to optimize.
            {'K562': 0, 'HepG2': 1, 'SKNSH': 2}.
        bending_factor: An optimization trick from the Boda paper.
        a_min: Min prediction output to clamp to. Paper always uses -2.0.
        a_max: Max prediction output to clamp to. Paper always uses 6.0
        target_alpha: Scaling factor for the bias term. Paper always uses 1.0

    Returns:
        torch.Tensor: Computed energy values.

    """
    output = model_output_tensor
    assert output.ndim == 2, output.shape
    assert 0 <= target_feature and target_feature < output.shape[-1]

    if a_min and a_max:
        output = output.clamp(min=a_min, max=a_max)
    elif a_min:
        output = output.clamp(min=a_min)
    elif a_max:
        output = output.clamp(max=a_max)

    if bending_factor:
        output = _bend(output, bending_factor)

    not_target_idx = [idx for idx in range(output.shape[-1]) if idx != target_feature]
    offtarget_most_expressed = output[..., not_target_idx].max(-1).values
    ontarget_expressed = output[..., target_feature]
    energy = offtarget_most_expressed - ontarget_expressed.mul(target_alpha)

    return energy


if __name__ == "__main__":
    # Test with a real model.
    m = Malinois(
        model_artifact="/Users/joelshor/github/nucleobench/malinois_artifacts__20211113_021200__287348.tar.gz",
        target_feature=0,
        bending_factor=1.0,
    )
    print(m(["A" * 200, "C" * 200, "T" * 200]))
