"""Parent class for models."""

from typing import Any, Optional

import torch
import numpy as np

from nucleobench.common import constants
from nucleobench.common import attribution_lib_torch as att_lib
from nucleobench.common import string_utils

SequenceType = str

class ModelClass(object):
    
    @staticmethod
    def init_parser():
        raise ValueError('Not implemented.')
    
    @staticmethod
    def debug_init_args() -> dict[str, Any]:
        raise ValueError('Not implemented.')
    
    def __init__(self, model_fn: callable, start_sequence: SequenceType):
        raise NotImplementedError("Not implemented.")
    
    def __call__(self, x: SequenceType, return_debug_info: bool) -> np.ndarray:
        """Takes in a string or list of strings, returns a scalar value per string."""
        raise NotImplementedError("Not implemented.")
    
    
class TISMModelClass(ModelClass):
    """Model that supports TISM."""
    
    def tism(self, x: str, idxs: Optional[list[int]] = None) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """Runs Taylor in-silico mutagenesis on inputs."""
        try:
            cur_vocab = self.vocab
        except AttributeError:
            cur_vocab = constants.VOCAB
            
        input_tensor = string_utils.dna2tensor(x, vocab_list=cur_vocab)
        sg_tensor = att_lib.grad_torch(
            input_tensor=torch.unsqueeze(input_tensor, dim=0),
            model=self.inference_on_tensor,
            idxs=idxs,
        )
        sg = att_lib.grad_tensor_to_dict(torch.squeeze(sg_tensor, dim=0), vocab=cur_vocab)
        x_effective = x if idxs is None else [x[idx] for idx in idxs]
        sg = att_lib.grad_to_tism(sg, x_effective)
        y = self.inference_on_tensor(torch.unsqueeze(input_tensor, dim=0))
        return y, sg
    
    
class PyTorchDifferentiableModel(ModelClass):
    """Model that can produce differentiable, PyTorch tensors."""
    
    def inference_on_tensor(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Not implemented.")