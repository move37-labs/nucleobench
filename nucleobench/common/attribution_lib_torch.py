"""Library for smoothgrad and genome-specific attribution methods.

Ref:
1. [Correcting gradient-based interpretations of deep neural networks for genomics](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-02956-3)
2. [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)
3. [Quick and effective approximation of in silico saturation mutagenesis experiments with first-order taylor expansion](https://pubmed.ncbi.nlm.nih.gov/39286491/)

To test locally:
```zsh
python -m nucleobench.common.attribution_lib
```
"""

import gc
from collections.abc import Callable

import torch

TISMOutputType = list[dict[str, float]]
SmoothgradVocabType = list[dict[str, torch.Tensor]]
TISMLocationsType = list[int]


def grad_torch(
    input_tensor: torch.Tensor,
    model: Callable[[torch.Tensor], torch.Tensor],
    idxs: TISMLocationsType | None = None,
    force_mem_clear: bool = False,
) -> torch.Tensor:
    """Generates gradients from a function.

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
    # Detach input to ensure we don't backprop into previous history.
    input_tensor = input_tensor.detach()

    # Run inference to get grads.
    if idxs is None:
        # In-place requires_grad is slightly faster/cleaner
        input_tensor.requires_grad_(True)
        x_grad = input_tensor
    else:
        input_tensor, x_grad = apply_gradient_mask(input_tensor, idxs)

    y = model(input_tensor)
    y.sum().backward(retain_graph=False)

    grads = x_grad.grad.detach().cpu()

    # Optional cleanup.
    del y
    del x_grad.grad

    if force_mem_clear:
        gc.collect()
        torch.cuda.empty_cache()

    assert grads.shape == x_grad.shape
    return grads


# TODO(joelshor): Add `attribution_lib.py` test, taken from `malinois/model_def_test.py`.
def grad_tensor_to_dict(
    smooth_grad: torch.Tensor, vocab: list[str]
) -> SmoothgradVocabType:
    """Map the smoothgrad indices to the vocab."""
    assert smooth_grad.ndim == 2
    assert list(smooth_grad.shape)[0] == len(vocab)

    def _to_dict(x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {vocab[i]: x[i] for i in range(len(vocab))}

    return [_to_dict(x) for x in smooth_grad.T]


def grad_to_tism(sg: SmoothgradVocabType, base_seq: str) -> TISMOutputType:
    """Returns result according to Taylor in-silico mutagenesis.

    Paper: https://www.cell.com/iscience/fulltext/S2589-0042(24)02032-7"""
    assert len(sg) == len(base_seq)

    tism = []
    for base_nt, sg_dict in zip(base_seq, sg):
        cur_tism = {}
        for nt, sg in sg_dict.items():
            if nt == base_nt:
                continue
            cur_tism[nt] = float(sg - sg_dict[base_nt])
        tism.append(cur_tism)

    return tism


def grad_torch_to_tism_torch(
    sg_tensor: torch.Tensor, base_seq: torch.Tensor
) -> torch.Tensor:
    """Returns result according to Taylor in-silico mutagenesis.

    Paper: https://www.cell.com/iscience/fulltext/S2589-0042(24)02032-7

    Identical to `smoothgrad_to_tism`, but for torch tensors. Avoids converting to strings.

    Args:
        sg_tensor: (vocab_size, seq_len) tensor, smoothgrad values for each base at each position.
        base_seq_onehot: (seq_len,) tensor, integer encoding of the reference sequence.
    Returns:
        tism_tensor: (vocab_size, seq_len) tensor, where for each position, the value for the reference base is zero,
        and for other bases is sg_tensor[nt, pos] - sg_tensor[ref_nt, pos].
    """
    assert sg_tensor.ndim == 2
    assert base_seq.ndim == 1
    assert sg_tensor.shape[1] == base_seq.shape[0]

    vocab_size, seq_len = sg_tensor.shape
    # Gather the smoothgrad value for the reference base at each position: (seq_len,)
    ref_vals = sg_tensor[base_seq, torch.arange(seq_len)]  # (seq_len,)
    # Expand to (vocab_size, seq_len) for broadcasting
    ref_vals_expanded = ref_vals.unsqueeze(0).expand(vocab_size, seq_len)
    # Subtract reference value from all
    tism_tensor = sg_tensor - ref_vals_expanded
    # Set the reference base positions to zero.
    # Not strictly necessary, but possibly relevant for numerical stability.
    tism_tensor[base_seq, torch.arange(seq_len)] = 0.0

    return tism_tensor


def apply_gradient_mask_deprecated(
    x: torch.Tensor, idxs: TISMLocationsType
) -> tuple[torch.Tensor, torch.Tensor]:
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
    tensor_slices = [
        x_grad[:, :, x_grad_i[i] : x_grad_i[i] + 1]
        if i in idxs
        else no_gradient[:, :, i : i + 1]
        for i in range(no_gradient.shape[2])
    ]
    x = torch.concat(tensor_slices, dim=2)

    return x, x_grad


def apply_gradient_mask(
    x: torch.Tensor, idxs: TISMLocationsType
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies a gradient mask by creating a computational graph where only 'idxs' are inputs.

    This effectively 'gathers' the values at idxs into a small tensor (x_grad),
    and 'scatters' them back into a static background to create the model input.
    """
    assert min(idxs) >= 0
    assert max(idxs) < x.shape[2]
    assert x.ndim == 3, x.shape
    assert idxs is not None

    # GATHER: Create the small leaf tensor for the gradients we actually want.
    # We use Ellipsis (...) to be agnostic to batch/channel dimensions.
    # Assuming x is (..., SequenceLength), and idxs indexes the last dim.
    # x_grad shape: (..., len(idxs))
    x_grad = x[..., idxs].detach().clone()
    x_grad.requires_grad_(True)

    # BACKGROUND: Create the full-sized static tensor.
    # This holds the values for positions we DON'T want to optimize.
    # It does not require gradients.
    model_input = x.detach().clone()

    # SCATTER: Insert the leaf tensor into the background.
    # This connects x_grad to the computation graph of 'model_input'.
    # In-place assignment [..., idxs] is differentiable and efficient in PyTorch.
    model_input[..., idxs] = x_grad

    # Return (Full Input for Model, Small Tensor for Gradients)
    return model_input, x_grad
