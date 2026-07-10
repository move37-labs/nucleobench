"""GFlowNet training core: environment, policy networks, training loop, and sampler.

This module is intentionally isolated from the nucleobench wrapper so the torchgfn
pieces can be swapped or extended independently.

Design:
- DNASequenceEnv: autoregressive construction of K editable positions within a
  fixed-length scaffold. Modeled directly after gfn.gym.discrete_ebm.DiscreteEBM.
  State: 1-D int tensor of length K (number of editable positions); -1 = unfilled.
  Actions 0..3: place VOCAB[a] at the next empty position (leftmost -1).
  Action 4: exit (only valid when all K slots are filled).
  Terminal states are spliced back into the scaffold to form full-length strings
  before passing to the oracle (reconstruct_full).
  When positions=range(L) the scaffold is fully overwritten and behaviour is
  identical to the original L-position construction.
- log_reward = -beta * energy, so the GFlowNet learns to sample low-energy sequences.
  This is identical in form to DiscreteEBM.log_reward = -alpha * energy.
"""

from collections.abc import Callable

import numpy as np
import torch
from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.estimators import DiscretePolicyEstimator
from gfn.gflownet import TBGFlowNet
from gfn.samplers import Sampler
from gfn.states import DiscreteStates, States
from gfn.utils.modules import MLP

from nucleobench.common import constants

VOCAB = constants.VOCAB
VOCAB_SIZE = len(VOCAB)  # 4
# Actions 0..VOCAB_SIZE-1: place nucleotide.  Action VOCAB_SIZE: exit.
N_ACTIONS = VOCAB_SIZE + 1
# Sentinel value for an unfilled position.
EMPTY = -1


class DNASequenceEnv(DiscreteEnv):
    """Autoregressive GFlowNet environment for DNA sequences with editable positions.

    The GFlowNet builds K nucleotides (one per editable position). Terminal states
    are spliced back into the frozen scaffold at `positions` to produce full-length
    strings for the oracle. When positions = range(len(scaffold)) the scaffold is
    fully overwritten and behaviour is identical to a plain length-L construction.

    Args:
        scaffold: The full-length reference sequence (used for frozen positions).
        positions: Sorted list of indices into scaffold that the GFlowNet may edit.
        model_fn: Black-box oracle, model_fn(list[str]) -> list[float].
            Called only on complete (terminal) states via full-length reconstructed seqs.
        beta: Reward temperature. log_reward = -beta * energy.
        vocab: Nucleotide alphabet, default VOCAB = ["A","C","G","T"].
    """

    def __init__(
        self,
        scaffold: str,
        positions: list[int],
        model_fn: Callable,
        beta: float = 2.0,
        vocab: list[str] = VOCAB,
    ):
        self.scaffold = scaffold
        self.positions = list(positions)
        self.build_len = len(positions)  # K: number of positions the GFlowNet fills
        self.model_fn = model_fn
        self.beta = beta
        self.vocab = vocab

        s0 = torch.full((self.build_len,), EMPTY, dtype=torch.long)
        # sf must differ from s0 and all valid terminal states.
        sf = torch.full((self.build_len,), VOCAB_SIZE + 1, dtype=torch.long)

        super().__init__(
            n_actions=N_ACTIONS,
            s0=s0,
            state_shape=(self.build_len,),
            sf=sf,
        )

    def reconstruct_full(self, state_tensor: torch.Tensor) -> list[str]:
        """Splice generated bases into the frozen scaffold at `positions`.

        Args:
            state_tensor: shape (*batch, build_len) with values in 0..VOCAB_SIZE-1.
                All positions must be filled (no -1); raises AssertionError otherwise
                to surface masking bugs loudly rather than silently emitting wrong seqs.

        Returns:
            List of full-length DNA strings (len = len(scaffold)).
        """
        flat = state_tensor.reshape(-1, self.build_len).cpu().numpy()
        assert (flat >= 0).all(), (
            "reconstruct_full received an unfilled state — a trajectory exited "
            "before all K slots were filled (masking bug)."
        )
        vocab_arr = np.array(self.vocab)
        base = list(self.scaffold)
        out = []
        for row in flat:
            s = base.copy()
            for j, pos in enumerate(self.positions):
                s[pos] = vocab_arr[row[j]]
            out.append("".join(s))
        return out

    # ------------------------------------------------------------------
    # States class with correct 2.4.1 masking API
    # ------------------------------------------------------------------

    def make_states_class(self) -> type[DiscreteStates]:
        """Returns a custom DiscreteStates subclass with correct forward/backward masks."""
        env = self

        class DNAStates(DiscreteStates):
            state_shape = (env.build_len,)
            s0 = env.s0
            sf = env.sf
            make_random_states = env.make_random_states
            n_actions = env.n_actions

            def _compute_forward_masks(self) -> torch.Tensor:
                """Forward masks: placement actions valid only at empty slots,
                exit valid only when fully filled."""
                batch = self.batch_shape
                masks = torch.zeros(
                    (*batch, env.n_actions),
                    dtype=torch.bool,
                    device=self.device,
                )
                has_empty = (self.tensor == EMPTY).any(dim=-1)  # (*batch,)
                # All 4 placement actions are allowed whenever there is an empty slot.
                masks[..., :VOCAB_SIZE] = has_empty.unsqueeze(-1).expand(
                    *batch, VOCAB_SIZE
                )
                # Exit action is only valid when the sequence is complete.
                masks[..., VOCAB_SIZE] = ~has_empty
                return masks

            def _compute_backward_masks(self) -> torch.Tensor:
                """Backward masks: can undo the last-placed nucleotide
                (rightmost non-empty slot)."""
                batch = self.batch_shape
                masks = torch.zeros(
                    (*batch, env.n_actions - 1),
                    dtype=torch.bool,
                    device=self.device,
                )
                last_pos = (self.tensor != EMPTY).sum(dim=-1) - 1  # (*batch,)
                valid = last_pos >= 0  # (*batch,)
                last_pos_clamped = last_pos.clamp(min=0)
                last_nt = self.tensor.gather(
                    -1, last_pos_clamped.unsqueeze(-1)
                ).squeeze(-1)  # (*batch,)
                for nt in range(VOCAB_SIZE):
                    masks[..., nt] = valid & (last_nt == nt)
                return masks

        return DNAStates

    # ------------------------------------------------------------------
    # Random states helper (used by make_states_class factory)
    # ------------------------------------------------------------------

    def make_random_states(
        self,
        batch_shape: tuple,
        conditions: torch.Tensor | None = None,
        device: torch.device | None = None,
        debug: bool = False,
    ) -> DiscreteStates:
        device = self.device if device is None else device
        tensor = torch.randint(
            EMPTY, VOCAB_SIZE, (*batch_shape, self.build_len), device=device
        )
        return self.States(tensor)

    # ------------------------------------------------------------------
    # Forward / backward transitions
    # ------------------------------------------------------------------

    def step(self, states: States, actions: Actions) -> States:
        """Place the chosen nucleotide at the leftmost empty (-1) position."""
        new_tensor = states.tensor.clone()
        action_vals = actions.tensor.squeeze(-1)  # (*batch,)

        is_empty = new_tensor == EMPTY
        leftmost_empty = is_empty.long().argmax(dim=-1)

        is_placement = action_vals < VOCAB_SIZE
        idx = leftmost_empty.unsqueeze(-1)  # (*batch, 1)
        new_tensor = torch.where(
            is_placement.unsqueeze(-1),
            new_tensor.scatter(-1, idx, action_vals.unsqueeze(-1)),
            new_tensor,
        )
        return self.States(new_tensor)

    def backward_step(self, states: States, actions: Actions) -> States:
        """Remove the rightmost filled position (set it back to -1)."""
        new_tensor = states.tensor.clone()
        is_filled = new_tensor != EMPTY
        rightmost_filled = self.build_len - 1 - is_filled.flip(-1).long().argmax(dim=-1)
        new_tensor.scatter_(-1, rightmost_filled.unsqueeze(-1), EMPTY)
        return self.States(new_tensor)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def log_reward(self, final_states: DiscreteStates) -> torch.Tensor:
        """log R(x) = -beta * energy(x).

        Reconstructs full-length sequences from terminal build states (splicing
        generated bases into the scaffold at `positions`) before calling model_fn.

        Args:
            final_states: terminal states (all build positions filled).

        Returns:
            Tensor of shape (*batch_shape,) with log-rewards.
        """
        seqs = self.reconstruct_full(final_states.tensor)
        energies = self.model_fn(seqs)
        if not isinstance(energies, torch.Tensor):
            energies = torch.tensor(energies, dtype=torch.float32)
        log_r = -self.beta * energies.to(final_states.tensor.device)
        assert log_r.shape == final_states.batch_shape
        return log_r


# ------------------------------------------------------------------
# Build GFlowNet (policy nets + TBGFlowNet)
# ------------------------------------------------------------------


def build_gflownet(
    env: DNASequenceEnv,
    hidden_dim: int = 128,
) -> tuple[TBGFlowNet, Sampler]:
    """Construct a TBGFlowNet with MLP forward and backward policy estimators.

    The forward policy MLP maps build-state vectors of length build_len (K) to
    logits over N_ACTIONS actions. For masked tasks this is K << L, keeping the
    network compact regardless of full sequence length.

    Args:
        env: The DNASequenceEnv to train on.
        hidden_dim: Width of MLP hidden layers.

    Returns:
        (gflownet, sampler) ready for training.
    """
    input_dim = env.build_len

    module_pf = MLP(
        input_dim=input_dim,
        output_dim=env.n_actions,
        hidden_dim=hidden_dim,
        n_hidden_layers=2,
    )
    module_pb = MLP(
        input_dim=input_dim,
        output_dim=env.n_actions - 1,
        hidden_dim=hidden_dim,
        n_hidden_layers=2,
        trunk=module_pf.trunk,
    )

    pf_estimator = DiscretePolicyEstimator(
        module=module_pf,
        n_actions=env.n_actions,
        is_backward=False,
    )
    pb_estimator = DiscretePolicyEstimator(
        module=module_pb,
        n_actions=env.n_actions,
        is_backward=True,
    )

    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0)
    sampler = Sampler(estimator=pf_estimator)
    return gflownet, sampler


# ------------------------------------------------------------------
# Sampling helper
# ------------------------------------------------------------------


def sample_sequences(
    sampler: Sampler,
    env: DNASequenceEnv,
    n: int,
) -> list[str]:
    """Sample n full-length sequences from the current policy.

    Args:
        sampler: On-policy sampler wrapping the forward policy estimator.
        env: The DNA sequence environment.
        n: Number of sequences to sample.

    Returns:
        List of n full-length DNA strings (len = len(env.scaffold)).
    """
    with torch.no_grad():
        trajectories = sampler.sample_trajectories(env=env, n=n, save_logprobs=False)
    terminal = trajectories.terminating_states
    return env.reconstruct_full(terminal.tensor)

