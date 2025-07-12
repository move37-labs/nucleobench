"""Custom implementation of Fast SeqProp."""

from typing import Optional

import argparse
import numpy as np
import torch
import tqdm

from nucleobench.common import constants
from nucleobench.common import string_utils
from nucleobench.common import testing_utils

from nucleobench.optimizations.typing import PositionsToMutateType, SequenceType, SamplesType, PyTorchDifferentiableModel
from nucleobench.optimizations import optimization_class as oc

from nucleobench.optimizations.fastseqprop_torch import fs_torch_module as fs_opt


class FastSeqProp(torch.nn.Module, oc.SequenceOptimizer):
    """Custom implementation of Fast SeqProp.
    Original paper [here](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04437-5).

    Other implementations:

    1. From the authors: [here](https://github.com/johli/seqprop/)
    1. From boda2: [here](https://github.com/sjgosai/boda2)"""
    
    def __init__(self, 
                 model_fn: PyTorchDifferentiableModel, 
                 start_sequence: SequenceType,
                 positions_to_mutate: Optional[PositionsToMutateType] = None,
                 learning_rate: float = 0.5,
                 eta_min: float = 1e-6,
                 vocab: list[str] = constants.VOCAB,
                 rnd_seed: int = 10,
                 ):
        torch.nn.Module.__init__(self)
        torch.manual_seed(rnd_seed)
        
        self.rnd_seed = rnd_seed
        self.vocab = vocab
        self.model_fn = model_fn
        self.reset(start_sequence, positions_to_mutate)
        
        self.learning_rate = learning_rate
        self.eta_min = eta_min
        
        # Test that model_fn is PyTorch, and accepts PyTorch tensors.
        # TODO(joelshor): Consider checking that the callable is a torch.nn.Module.
        ret = self.model_fn.inference_on_tensor(self.get_samples_tensor(n_samples=2))
        if not isinstance(ret, torch.Tensor):
            raise ValueError('FastSeqProp model must be pytorch.')
        
    def reset(self, seq: SequenceType, positions_to_mutate: Optional[list[int]] = None):
        self.start_sequence = seq
        cur_tensor = string_utils.dna2tensor(seq, vocab_list=self.vocab)
        cur_tensor = torch.unsqueeze(cur_tensor, dim=0)
        assert cur_tensor.ndim == 3
        
        self.opt_module = fs_opt.TorchFastSeqPropOptimizer(
            start_tensor=cur_tensor,
            positions_to_mutate=positions_to_mutate,
            vocab_len=4,
            tau=1.0,
        )
        
    
    def energy(self, batch_size: int) -> torch.Tensor:
        """Energy on current params."""
        sampled_nts_onehot = self.opt_module.get_samples_onehot(batch_size)
        return self.model_fn.inference_on_tensor(sampled_nts_onehot)
        
        
    @staticmethod
    def run_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        group = parser.add_argument_group('FastSeqprop run args')
        
        group.add_argument('--batch_size', type=int, default=256, required=True, help='')
        
        return parser
    
    @staticmethod
    def debug_run_args():
        return {
            'batch_size': 2,
        }
        
    def run(self, 
            n_steps: int,
            batch_size: int,
            ) -> list[np.ndarray]:
        """Runs the optimization.
        
        Default hparams come from https://www.nature.com/articles/s41586-024-08070-z.
        """
        assert len(list(self.opt_module.parameters())) == 1
        only_param = list(self.opt_module.parameters())[0]
        
        optimizer = torch.optim.Adam([only_param], lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_steps, eta_min=self.eta_min)
        
        energies = []
        for _ in tqdm.tqdm(range(n_steps)):
            optimizer.zero_grad()
            energy = self.energy(batch_size).double()
            assert list(energy.shape) == [batch_size]
            
            energies.append(energy.detach().cpu().numpy())
            energy = energy.mean()
            energy.backward(inputs=[only_param])
            optimizer.step()
            scheduler.step()
        return energies
        
    
    def get_samples_tensor(self, n_samples: int) -> torch.Tensor:
        return self.opt_module.get_samples_onehot(n_samples)
        
    
    def get_samples(self, n_samples: int) -> SamplesType:
        """Get samples."""
        samples_onehot = self.get_samples_tensor(n_samples)
        assert samples_onehot.ndim == 3
        assert list(samples_onehot.shape[0:2]) == [n_samples, len(self.vocab)]
        
        all_ret = []
        for cur_tensor in samples_onehot:
            cur_str = ''
            for onehot_nt in cur_tensor.T:
                assert onehot_nt.sum() == 1
                nonzeros = onehot_nt.nonzero()
                assert len(nonzeros) == 1
                idx = nonzeros[0]
                cur_str += self.vocab[idx]
            all_ret.append(cur_str)
        return all_ret
    
    def is_finished(self) -> bool:
        return False
    
    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        group = parser.add_argument_group('FastSeqprop init args')
        
        group.add_argument('--learning_rate', type=float, default=0.5, required=True, help='')
        group.add_argument('--eta_min', type=float, default=1e-6, required=True, help='')
        group.add_argument('--rnd_seed', type=int, default=10, required=False, help='')
        
        return parser
    
    @staticmethod
    def debug_init_args():
        return {
            'model_fn': testing_utils.CountLetterModel(),
            'start_sequence': 'AA',
            'positions_to_mutate': [1],
            'rnd_seed': 0,
        }