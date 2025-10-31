"""Sets up a torch NN for optimization of the input."""

from typing import Optional

import torch

class TorchFastSeqPropOptimizer(torch.nn.Module):
    def __init__(self, 
                 start_logits: torch.Tensor,
                 positions_to_mutate: Optional[list[int]] = None,
                 vocab_len: int = 4,
                 tau: float = 1.0,
                 use_norm: bool = False,
                 use_slope_annealing: bool = True,
                 ):
        super().__init__()
        
        assert start_logits.ndim == 3
        assert start_logits.shape[1] == vocab_len
        
        self.use_norm = use_norm
        self.use_slope_annealing = use_slope_annealing
        
        if positions_to_mutate is None:
            self.gradient_mask = None
        else:
            # Change logits to be deterministic.
            inverse_pos_to_mask = torch.ones(start_logits.shape[2], dtype=torch.bool)
            inverse_pos_to_mask[positions_to_mutate] = False
            
            top_bp = start_logits.argmax(dim=1, keepdim=True)  
            start_logits[:, :, inverse_pos_to_mask] = -10**9  # Assign negative inf
            sliced_logits = start_logits[:, :, inverse_pos_to_mask] # 1. Get the slice (this is a copy)
            sliced_indices = top_bp[:, :, inverse_pos_to_mask] # 2. Get the indices corresponding to that slice
            scattered_slice = sliced_logits.scatter(dim=1, index=sliced_indices, value=10**9)  # 3. Perform the scatter (use the non-in-place version)
            start_logits[:, :, inverse_pos_to_mask] = scattered_slice  # 4. Assign the new tensor back to the original tensor's slice
            
            # Set up gradient mask.
            self.gradient_mask = torch.zeros_like(start_logits)
            self.gradient_mask[:, :, positions_to_mutate] = 1
            
        self.register_parameter(
            'params', torch.nn.Parameter(start_logits.detach().clone()))
        
        if self.use_norm:
            self.normalization = torch.nn.InstanceNorm1d(
                num_features=vocab_len, 
                affine=False)
        self.vocab_len = vocab_len
        self.tau = tau
        
    
    def get_logits(self) -> torch.Tensor:
        if self.gradient_mask is None:
            params_eff = self.params
        else:
            params_eff = self.mask_gradients(self.params)
            
        if self.use_norm:
            return self.normalization(params_eff) / self.tau
        else:
            return params_eff / self.tau
    
    def get_probs(self):
        return torch.nn.functional.softmax(self.get_logits(), dim=1)
    
    def get_samples_onehot(self, n_samples) -> torch.Tensor:
        """Draw samples.
        
        For now, assume that the patch dimension of the parameter is 1.
        
        TODO(joelshor): Expand to multiple batches, if desired.
        TODO(joelshor): Switch to using logits instead of probs.
        """
        # TODO(joelshor): Consider using logits instead of probs.
        probs = self.get_probs()
        assert probs.ndim == 3
        assert probs.shape[0] == 1
        assert probs.shape[1] == self.vocab_len
        seq_len = probs.shape[2]
        
        # For now, remove ability to sample from batches.
        probs = torch.squeeze(probs, dim=0)
        
        sampled_idxs = torch.distributions.categorical.Categorical(probs.T)
        samples = sampled_idxs.sample( (n_samples, ) )
        assert list(samples.shape) == [n_samples, seq_len]
        samples_onehot = torch.nn.functional.one_hot(samples, num_classes=self.vocab_len)
        
        if self.use_slope_annealing:
            # Apply the "slope annealing trick", as described in https://arxiv.org/pdf/1609.01704
            # and used in https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04437-5.
            trick_factor = probs.T.repeat(n_samples, 1, 1)
            samples_onehot = samples_onehot - trick_factor.detach() + trick_factor
            
        samples_onehot = samples_onehot.permute(0, 2, 1)
        assert list(samples_onehot.shape) == [n_samples, self.vocab_len, seq_len]
        return samples_onehot
    
    
    def mask_gradients(self, x: torch.Tensor) -> torch.Tensor:
        assert self.gradient_mask is not None
        
        grad_pass  = x.mul(self.gradient_mask)
        grad_block = x.detach().mul(1 - self.gradient_mask)
        
        return grad_pass + grad_block