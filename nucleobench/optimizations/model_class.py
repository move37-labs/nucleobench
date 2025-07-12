"""Parent class for models."""

from typing import Any, Optional

import torch
import numpy as np

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
    
    def tism(self, x: str, idx: Optional[int] = None) -> tuple[np.ndarray, Any]:
        """Returns inference and TISM for a single input."""
        raise NotImplementedError("Not implemented.")
    
    
class PyTorchDifferentiableModel(ModelClass):
    """Model that can produce differentiable, PyTorch tensors."""
    
    def inference_on_tensor(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Not implemented.")