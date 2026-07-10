"""GFlowNet sequence optimizer for nucleobench."""

import argparse

import numpy as np
import torch
import tqdm

from nucleobench.common import constants, testing_utils
from nucleobench.optimizations import optimization_class as oc
from nucleobench.optimizations.gflownet import gflownet_core as core
from nucleobench.optimizations.typing import (
    ModelType,
    PositionsToMutateType,
    SamplesType,
    SequenceType,
)


class GFlowNet(oc.SequenceOptimizer):
    """GFlowNet-based sequence designer (Trajectory Balance, discrete autoregressive).

    Generates sequences by training a GFlowNet to sample from a Boltzmann
    distribution proportional to exp(-beta * model_fn(x)). Lower oracle energy
    corresponds to higher GFlowNet reward, so training drives the policy toward
    low-energy (high-quality) sequences.

    Reference: Bengio et al. "Flow Network based Generative Models for
    Non-Iterative Diverse Candidate Generation" (NeurIPS 2021).
    """

    def __init__(
        self,
        model_fn: ModelType,
        start_sequence: SequenceType,
        positions_to_mutate: PositionsToMutateType | None = None,
        beta: float = 2.0,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        hidden_dim: int = 128,
        rnd_seed: int = 0,
        vocab: list[str] = constants.VOCAB,
    ):
        torch.manual_seed(rnd_seed)
        np.random.seed(rnd_seed)

        self.model_fn = model_fn
        self.start_sequence = start_sequence
        self.seq_len = len(start_sequence)
        self.vocab = vocab
        self.beta = beta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.rnd_seed = rnd_seed

        self.positions = (
            list(range(self.seq_len))
            if positions_to_mutate is None
            else sorted(set(positions_to_mutate))
        )

        self._env = core.DNASequenceEnv(
            scaffold=start_sequence,
            positions=self.positions,
            model_fn=model_fn,
            beta=beta,
            vocab=vocab,
        )
        self._gflownet, self._sampler = core.build_gflownet(
            self._env,
            hidden_dim=hidden_dim,
        )

        # Track the best (lowest-energy) sequence seen during training.
        self._best_sequence: SequenceType = start_sequence
        self._best_energy: float = float("inf")

    # ------------------------------------------------------------------
    # SequenceOptimizer interface
    # ------------------------------------------------------------------

    def run(self, n_steps: int) -> list[np.ndarray]:
        """Train for n_steps iterations; each iteration samples a batch, computes
        the Trajectory Balance loss, and performs one optimizer step.

        Args:
            n_steps: Number of training iterations.

        Returns:
            List of n_steps arrays, each of shape (batch_size,), containing the
            per-sample oracle energies for that iteration's batch.
        """
        optimizer = torch.optim.Adam(
            self._gflownet.pf_pb_parameters(), lr=self.learning_rate
        )
        optimizer.add_param_group(
            {"params": self._gflownet.logz_parameters(), "lr": self.learning_rate * 10}
        )

        energies_per_step: list[np.ndarray] = []

        for _ in tqdm.tqdm(range(n_steps)):
            trajectories = self._sampler.sample_trajectories(
                env=self._env,
                n=self.batch_size,
                save_logprobs=True,
            )
            optimizer.zero_grad()
            loss = self._gflownet.loss(self._env, trajectories)
            loss.backward()
            optimizer.step()

            # Compute per-sample oracle energies for the current batch.
            terminal = trajectories.terminating_states
            seqs = self._env.reconstruct_full(terminal.tensor)
            raw_energies = self.model_fn(seqs)
            if not isinstance(raw_energies, np.ndarray):
                raw_energies = np.array(raw_energies, dtype=np.float32)

            energies_per_step.append(raw_energies)

            # Update best sequence.
            batch_best_idx = int(np.argmin(raw_energies))
            if raw_energies[batch_best_idx] < self._best_energy:
                self._best_energy = float(raw_energies[batch_best_idx])
                self._best_sequence = seqs[batch_best_idx]

        return energies_per_step

    def get_samples(self, n_samples: int) -> SamplesType:
        """Return n_samples sequences from the trained policy.

        The first entry is always the best (lowest-energy) sequence seen during
        training, mirroring fs.get_samples behaviour. The remaining entries are
        fresh stochastic draws from the current policy.

        Args:
            n_samples: Total number of sequences to return.

        Returns:
            List of n_samples DNA strings, each of length seq_len.
        """
        best = [self._best_sequence]
        if n_samples == 1:
            return best
        stochastic = core.sample_sequences(self._sampler, self._env, n_samples - 1)
        return best + stochastic

    def is_finished(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Argparse / debug interface
    # ------------------------------------------------------------------

    @staticmethod
    def init_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="", add_help=False)
        group = parser.add_argument_group("GFlowNet init args")

        group.add_argument(
            "--beta",
            type=float,
            default=2.0,
            required=False,
            help="Reward temperature. log_reward = -beta * energy.",
        )
        group.add_argument(
            "--learning_rate",
            type=float,
            default=1e-3,
            required=False,
            help="Adam learning rate for policy parameters.",
        )
        group.add_argument(
            "--batch_size",
            type=int,
            default=32,
            required=False,
            help="Number of trajectories per training iteration.",
        )
        group.add_argument(
            "--hidden_dim",
            type=int,
            default=128,
            required=False,
            help="Hidden dimension of the policy MLP.",
        )
        group.add_argument(
            "--rnd_seed",
            type=int,
            default=0,
            required=False,
            help="Random seed for torch and numpy.",
        )

        return parser

    @staticmethod
    def debug_init_args() -> dict:
        return {
            "model_fn": testing_utils.CountLetterModel(),
            "start_sequence": "ACGT",
            "rnd_seed": 0,
            "beta": 2.0,
            "learning_rate": 1e-3,
            "batch_size": 8,
            "hidden_dim": 32,
        }
