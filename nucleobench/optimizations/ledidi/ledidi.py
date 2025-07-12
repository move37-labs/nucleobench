"""Wrapper around gRelu Ledidi."""

from typing import Optional

import argparse
import numpy as np
import torch

from nucleobench.common import constants
from nucleobench.common import string_utils
from nucleobench.common import testing_utils

from nucleobench.optimizations.typing import PositionsToMutateType, SequenceType, SamplesType, PyTorchDifferentiableModel
from nucleobench.optimizations import optimization_class as oc

from nucleobench.optimizations.ledidi import ledidi_module as ledidi


class Ledidi(oc.SequenceOptimizer):
    """Wrapper around Ledidi, inspired by gRelu.
    Original paper [here](https://www.biorxiv.org/content/10.1101/2020.05.21.109686v1.full).
    """
    
    def __init__(self, 
                 model_fn: PyTorchDifferentiableModel, 
                 start_sequence: SequenceType,
                 positions_to_mutate: Optional[PositionsToMutateType] = None,
                 vocab: list[str] = constants.VOCAB,
                 # Defaults taken from Ledidi paper.
                 train_batch_size: int = 64, 
                 lr: float = 0.1,
                 use_input_loss: bool = False,
                 rng_seed: int = 0,
                 debug: bool = False,
                 ):
        torch.manual_seed(rng_seed)
        np.random.seed(rng_seed)
        
        self.vocab = vocab
        self.start_sequence = start_sequence
        
        self.positions_to_mutate = positions_to_mutate
        self.train_batch_size = train_batch_size
        self.lr = lr
        self.use_input_loss = use_input_loss
        
        # Convert sequence into a one-hot encoded tensor.
        self.seed_tensor = string_utils.dna2tensor(self.start_sequence)
        if isinstance(model_fn, torch.nn.Module):
            self.model_fn = model_fn.eval()
            for param in self.model_fn.parameters():
                param.requires_grad = False
        else:
            self.model_fn = model_fn
            self.model_fn.model = self.model_fn.model.eval()
            for param in self.model_fn.model.parameters():
                param.requires_grad = False
        
        # Test that model_fn is PyTorch, and accepts PyTorch tensors.
        # TODO(joelshor): Consider checking that the callable is a torch.nn.Module.
        ret = self.model_fn.inference_on_tensor(torch.unsqueeze(self.seed_tensor, 0))
        if not isinstance(ret, torch.Tensor):
            raise ValueError('Ledidi model must be pytorch.')

        # Create input mask.
        if self.positions_to_mutate is not None:
            input_mask = torch.Tensor([False] * self.seed_tensor.shape[-1]).type(torch.bool)
            input_mask[self.positions_to_mutate] = True
        else:
            input_mask = None

        # Initializing ledid obj modeled after gRelu.
        def loss_func(x):
            # No need to cast as Tensor, or flip the sign. That should be taken
            # care of by the underlying model.
            return x.mean()
        
        # Initialize ledidi
        self.designer = ledidi.Ledidi(
            self.model_fn,
            self.seed_tensor.shape,
            output_loss=loss_func,
            max_iter=None,
            input_mask=input_mask,
            input_loss=torch.nn.L1Loss(reduction='sum') if self.use_input_loss else None,
            batch_size=self.train_batch_size,
            lr=self.lr,
            return_history=True,
            verbose=debug,
        )
        
        
    @staticmethod
    def run_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        return parser
    
    @staticmethod
    def debug_run_args():
        return {
        }
        
    def run(self, 
            n_steps: int,
            ) -> list[np.ndarray]:
        """Runs the optimization."""
        self.designer.max_iter = n_steps
        
        assert self.designer.batch_size == self.train_batch_size
        _, history = self.designer.fit_transform(
            torch.unsqueeze(self.seed_tensor, 0))
        
        return history['output_loss']
        
    
    def get_samples(self, n_samples: int) -> SamplesType:
        """Get samples."""
        prev_bs = self.designer.batch_size
        self.designer.batch_size = n_samples
        X_hat = self.designer(torch.unsqueeze(self.seed_tensor, 0))
        self.designer.batch_size = prev_bs
        return string_utils.tensor2dna_batch(X_hat.detach(), vocab_list=self.vocab)
    
    def is_finished(self) -> bool:
        return False
    
    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        group = parser.add_argument_group('Ledidi init args')
        
        group.add_argument('--train_batch_size', type=int, default=256, required=True, help='')
        group.add_argument('--lr', type=float, default=0.1, required=True, help='')
        group.add_argument('--rng_seed', type=int, default=0, required=False, help='')
        
        return parser
    
    @staticmethod
    def debug_init_args():
        return {
            'model_fn': testing_utils.CountLetterModel(),
            'start_sequence': 'AA',
            'train_batch_size': 4,
            'lr': 0.1,
            'rng_seed': 0,
        }