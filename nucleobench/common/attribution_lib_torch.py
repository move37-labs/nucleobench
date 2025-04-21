"""Library for smoothgrad and genome-specific attribution methods.

Ref:
1. [Correcting gradient-based interpretations of deep neural networks for genomics](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-02956-3)
2. [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)
3. [Quick and effective approximation of in silico saturation mutagenesis experiments with first-order taylor expansion](https://pubmed.ncbi.nlm.nih.gov/39286491/)

TODO(joelshor): Consider using public version of attribution tools, such as:
- PyTorch Smoothgrad: https://github.com/pkmr06/pytorch-smoothgrad
- PyTorch GradCam and others: https://github.com/jacobgil/pytorch-grad-cam?tab=readme-ov-file
- PyTorch Smoothgrad and others: https://tf-explain.readthedocs.io/en/latest/

To test locally:
```zsh
python -m nucleobench.common.attribution_lib
```
"""

import gc
import numpy as np
import torch
from typing import Callable, Optional


TISMOutputType = list[dict[str, float]]
SmoothgradVocabType = list[dict[str, torch.Tensor]]
TISMLocationsType = list[int]


def noise_inputs(
    input_tensor: torch.Tensor,
    noise_stdev: float,
    times: int,
    ) -> torch.Tensor:
    """Generates noisy inputs.

    NOTE: For simplicity, for now, we work with SINGLE TENSORS. Assume no batch dimension.

    Args:
        input_tensor: Input tensor. Doesn't have to be genomic. Should NOT be batched.
        noise_stdev: Noise to add.
        times: Number of times to add noise.
    """
    if noise_stdev < 0:
        raise ValueError(f'Requires non-negative noise stdev: {noise_stdev}')
    x = input_tensor  # Syntactic sugar.
    
    # Stack N versions of the input, to add uncorrelated noise to.
    with torch.no_grad():
        x = x.unsqueeze(0)
        x = x.repeat([times] + [1] * (x.ndim-1))

        # Add noise for hte smoothgrad algorithm.
        if noise_stdev > 0:
            noise_to_add = torch.normal(mean=torch.zeros(x.shape), std=noise_stdev)
            x += noise_to_add
    return x

def noisy_grads_torch(
        input_tensor: torch.Tensor, 
        model: Callable[[torch.Tensor], torch.Tensor], 
        noise_stdev: float, 
        times: int,
        idxs: Optional[TISMLocationsType] = None,
        ) -> torch.Tensor:
    """Generates noisy gradients from a function.

    NOTE: For simplicity, for now, we work with SINGLE TENSORS. Assume no batch dimension.

    This replicates the input `times` times, and runs it through the network all at once.

    TODO(joelshor): Add batching, for the situation where `times` is larger than the possible batch size
        of a single inference with a network.
    TODO(joelshor): Add ability to efficiently compute multiple inputs at once.

    Args:
        input_tensor (torch.Tensor): Input tensor. Doesn't have to be genomic. Should NOT be batched.
        model: PyTorch model to use. The model must return a scalar per batch element.
        noise_stdev: Noise to add.
        times: Number of times to add noise.
        idx: If present, only backprop through this location.
    """
    x = noise_inputs(
        input_tensor=input_tensor,
        noise_stdev=noise_stdev,
        times=times)

    # Run inference to get grads.
    if idxs is None:
        x_grad = x
        x_grad.requires_grad = True
    else:
        x, x_grad = apply_gradient_mask(x, idxs)
        
    y = model(x)
    y_sum = y.sum()
    y_sum.backward(retain_graph=False)
    noisy_grads = x_grad.grad.numpy()

    gc.collect()
    torch.cuda.empty_cache()

    assert noisy_grads.shape == x_grad.shape
    return noisy_grads


def smoothgrad_torch(
    input_tensor: torch.Tensor, 
    model: Callable[[torch.Tensor], torch.Tensor], 
    noise_stdev: float, 
    times: int,
    idxs: Optional[TISMLocationsType] = None,
    ) -> torch.Tensor:
    """Custom implementation of SmoothGrad.
    https://arxiv.org/pdf/1706.03825

    NOTE: For simplicity, for now, we work with SINGLE TENSORS. Assume no batch dimension.

    This replicates the input `times` times, and runs it through the network all at once.

    TODO(joelshor): Add batching, for the situation where `times` is larger than the possible batch size
        of a single inference with a network.
    TODO(joelshor): Add ability to efficiently compute multiple inputs at once.

    Args:
        input_tensor (torch.Tensor): Input tensor. Doesn't have to be genomic. Should NOT be batched.
        model: PyTorch model to use. The model must return a scalar per batch element.
        noise_stdev: Noise to add.
        times: Number of times to add noise.
        idx: If present, only backprop to this location.
        
    Returns:
        Per-nucleotide smoothgrad.
    """
    noisy_grads = noisy_grads_torch(
        input_tensor=input_tensor,
        model=model,
        noise_stdev=noise_stdev,
        times=times,
        idxs=idxs,
    )

    return np.mean(noisy_grads, axis=0)


# TODO(joelshor): Add `attribution_lib.py` test, taken from `malinois/model_def_test.py`.
def smoothgrad_tensor_to_dict(smooth_grad: torch.Tensor, vocab: list[str]) -> SmoothgradVocabType:
    """Map the smoothgrad indices to the vocab."""
    assert smooth_grad.ndim == 2
    assert list(smooth_grad.shape)[0] == len(vocab)
    def _to_dict(x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {vocab[i]: x[i] for i in range(len(vocab))}
    return [_to_dict(x) for x in smooth_grad.T]


def smoothgrad_to_tism(sg: SmoothgradVocabType, base_seq: str) -> TISMOutputType:
    """Returns result according to Taylor in-silico mutagenesis.
    
    Paper: https://www.cell.com/iscience/fulltext/S2589-0042(24)02032-7"""
    assert len(sg) == len(base_seq)

    tism = []
    for base_nt, sg_dict in zip(base_seq, sg):
        cur_tism = {}
        for nt, sg in sg_dict.items():
            if nt == base_nt: continue
            cur_tism[nt] = float(sg - sg_dict[base_nt])
        tism.append(cur_tism)

    return tism


def apply_gradient_mask(x: torch.Tensor, idxs: TISMLocationsType) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies a gradient mask to the input tensor.
    
    NOTE: Do NOT just multiply by 0. This will run out of memory in large models.

    Returns:
        Tuple of (x, masked_x), where masked_x is the input tensor with the gradient mask applied.
    """
    assert min(idxs) >= 0
    assert max(idxs) < x.shape[2]
    assert x.ndim == 3, x.shape
    
    no_gradient = x.clone().detach()
    no_gradient.requires_grad = False
    
    x_grad = x[:, :, idxs].clone().detach()
    x_grad.requires_grad = True
    x_grad_i = {idx: i for i, idx in enumerate(idxs)}
    
    # Instead of using `torch.where`, we use this method to make our gradient tensor
    # as small as possible, to preserve memory.
    tensor_slices = [x_grad[:, :, x_grad_i[i]:x_grad_i[i]+1] if i in idxs
                     else no_gradient[:, :, i:i+1]
                     for i in range(no_gradient.shape[2])]
    x = torch.concat(tensor_slices, dim=2)
    
    return x, x_grad