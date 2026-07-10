"""Custom implementation of Fast SeqProp."""

import argparse

import numpy as np
import torch
import tqdm

from nucleobench.common import constants, string_utils, testing_utils
from nucleobench.optimizations import optimization_class as oc
from nucleobench.optimizations.typing import (
    PositionsToMutateType,
    PyTorchDifferentiableModel,
    SamplesType,
    SequenceType,
)

from . import fs_torch_module as fs_opt


class FastSeqProp(torch.nn.Module, oc.SequenceOptimizer):
    """Custom implementation of Fast SeqProp.
    Original paper [here](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04437-5).

    Other implementations:

    1. From the authors: [here](https://github.com/johli/seqprop/)
    1. From boda2: [here](https://github.com/sjgosai/boda2)"""

    def __init__(
        self,
        model_fn: PyTorchDifferentiableModel,
        start_sequence: SequenceType,
        learning_rate: float,
        batch_size: int,
        eta_min: float = 1e-6,
        positions_to_mutate: PositionsToMutateType | None = None,
        log_min: float = 0.25,
        vocab: list[str] = constants.VOCAB,
        rnd_seed: int = 10,
    ):
        torch.nn.Module.__init__(self)
        torch.manual_seed(rnd_seed)

        self.rnd_seed = rnd_seed
        self.vocab = vocab
        self.model_fn = model_fn
        self.log_min = log_min
        self.reset(start_sequence, positions_to_mutate)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.eta_min = eta_min

        # Test that model_fn is PyTorch, and accepts PyTorch tensors.
        # TODO(joelshor): Consider checking that the callable is a torch.nn.Module.
        ret = self.model_fn.inference_on_tensor(self.get_samples_tensor(n_samples=2))
        if not isinstance(ret, torch.Tensor):
            raise ValueError("FastSeqProp model must be pytorch.")

    def reset(self, seq: SequenceType, positions_to_mutate: list[int] | None = None):
        self.start_sequence = seq
        cur_onehot = string_utils.dna2tensor(seq, vocab_list=self.vocab)
        cur_onehot = torch.unsqueeze(cur_onehot, dim=0)
        assert cur_onehot.ndim == 3

        self.opt_module = fs_opt.TorchFastSeqPropOptimizer(
            start_probs=cur_onehot,
            positions_to_mutate=positions_to_mutate,
            vocab_len=len(self.vocab),
            log_min=self.log_min,
            tau=1.0,
        )

    def energy(self, batch_size: int) -> torch.Tensor:
        """Energy on current params."""
        sampled_nts_onehot = self.opt_module.get_samples_onehot(batch_size)
        ret = self.model_fn.inference_on_tensor(sampled_nts_onehot)
        assert isinstance(ret, torch.Tensor)
        return ret

    def run(self, n_steps: int) -> list[np.ndarray]:
        """Runs the optimization.

        Default hparams come from https://www.nature.com/articles/s41586-024-08070-z.
        """
        assert len(list(self.opt_module.parameters())) == 1
        only_param = list(self.opt_module.parameters())[0]

        optimizer = torch.optim.Adam([only_param], lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_steps, eta_min=self.eta_min
        )

        energies = []
        for _ in tqdm.tqdm(range(n_steps)):
            optimizer.zero_grad()
            energy = self.energy(self.batch_size).double()
            assert list(energy.shape) == [self.batch_size]

            energies.append(energy.detach().cpu().numpy())
            energy = energy.mean()
            energy.backward()
            optimizer.step()
            scheduler.step()
        return energies

    def get_samples_tensor(self, n_samples: int) -> torch.Tensor:
        return self.opt_module.get_samples_onehot(n_samples)

    def _decode_onehot(self, samples_onehot: torch.Tensor) -> SamplesType:
        """Decode a batch of one-hot tensors to a list of strings.

        Args:
            samples_onehot: shape (n, vocab_len, seq_len) with true one-hot entries.

        Returns:
            List of n decoded sequences.
        """
        assert samples_onehot.ndim == 3
        assert samples_onehot.shape[1] == len(self.vocab)
        all_ret = []
        for cur_tensor in samples_onehot:
            cur_str = ""
            for onehot_nt in cur_tensor.T:
                assert onehot_nt.sum() == 1
                nonzeros = onehot_nt.nonzero()
                assert len(nonzeros) == 1
                idx = nonzeros[0]
                cur_str += self.vocab[idx]
            all_ret.append(cur_str)
        return all_ret

    def get_best_sequence(self) -> SequenceType:
        """Return the single deterministic best sequence (argmax of the PWM).

        Takes the highest-probability nucleotide at every position. Matches the
        reference implementation's evaluation mode (`st_hardmax_softmax`) and
        removes stochasticity from final sequence extraction.
        """
        best_onehot = self.opt_module.get_best_onehot()  # (1, vocab_len, seq_len)
        return self._decode_onehot(best_onehot)[0]

    def get_samples(self, n_samples: int) -> SamplesType:
        """Get n_samples sequences from the current PWM.

        The first entry is always the deterministic argmax sequence (matching the
        reference implementation's eval mode). The remaining n_samples-1 entries
        are stochastic draws via the straight-through categorical sampler.
        """
        best = [self.get_best_sequence()]
        if n_samples == 1:
            return best
        stochastic = self._decode_onehot(self.get_samples_tensor(n_samples - 1))
        return best + stochastic

    def is_finished(self) -> bool:
        return False

    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        group = parser.add_argument_group("FastSeqprop init args")

        group.add_argument(
            "--learning_rate", type=float, default=0.5, required=True, help=""
        )
        group.add_argument("--eta_min", type=float, required=True, help="")
        group.add_argument("--rnd_seed", type=int, required=True, help="")
        group.add_argument("--batch_size", type=int, required=True, help="")

        return parser

    @staticmethod
    def debug_init_args():
        return {
            "model_fn": testing_utils.CountLetterModel(),
            "start_sequence": "AA",
            "rnd_seed": 0,
            "learning_rate": 0.5,
            "eta_min": 1e-6,
            "batch_size": 4,
            "log_min": 0.25,
        }
