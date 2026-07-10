"""GFlowNet training core: environment, policy networks, training loop, and sampler.

This module is intentionally isolated from the nucleobench wrapper so the torchgfn
pieces can be swapped or extended independently.

Design:
- DNASequenceEnv: autoregressive construction of a length-L DNA sequence over a
  4-letter vocab. Modeled directly after gfn.gym.discrete_ebm.DiscreteEBM.
  State: 1-D int tensor of length L; -1 means "not yet filled".
  Actions 0..3: place VOCAB[a] at the next empty position (leftmost -1).
  Action 4: exit (only valid when all positions are filled).
- log_reward = -beta * energy, so the GFlowNet learns to sample low-energy sequences.
  This is identical in form to DiscreteEBM.log_reward = -alpha * energy.
- The backward policy is fixed and uniform (constant_pb=True in TBGFlowNet), which
  is valid for an autoregressive DAG where every terminal state has exactly one parent
  chain of length L.
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
    """Autoregressive GFlowNet environment for fixed-length DNA sequences.

    States are 1-D int tensors of length `seq_len`. -1 marks unfilled positions.
    At each step the policy places one of the 4 nucleotides at the leftmost empty
    slot. Once all slots are filled the only valid action is exit.

    Args:
        seq_len: Length of sequences to generate.
        model_fn: Black-box oracle, model_fn(list[str]) -> list[float].
            Called only on complete (terminal) states.
        beta: Reward temperature. log_reward = -beta * energy.
    """

    def __init__(
        self,
        seq_len: int,
        model_fn: Callable,
        beta: float = 2.0,
    ):
        self.seq_len = seq_len
        self.model_fn = model_fn
        self.beta = beta

        s0 = torch.full((seq_len,), EMPTY, dtype=torch.long)
        # sf is the library's "sink" state used for padding; must differ from s0 and
        # all valid terminal states. We use a value outside {-1, 0, 1, 2, 3}.
        sf = torch.full((seq_len,), VOCAB_SIZE + 1, dtype=torch.long)

        super().__init__(
            n_actions=N_ACTIONS,
            s0=s0,
            state_shape=(seq_len,),
            sf=sf,
        )

    # ------------------------------------------------------------------
    # States class with correct 2.4.1 masking API
    # ------------------------------------------------------------------

    def make_states_class(self) -> type[DiscreteStates]:
        """Returns a custom DiscreteStates subclass with correct forward/backward masks."""
        env = self

        class DNAStates(DiscreteStates):
            state_shape = (env.seq_len,)
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
                # (The env.step logic always fills the leftmost slot, so only one slot
                # is targeted per step; allowing all 4 nucleotide choices is correct.)
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
                # Any of the 4 nucleotide placements could have been the last action.
                # The backward mask just needs to be non-zero at the actions that
                # could have led to this state. Since we track position implicitly
                # (always fill leftmost), any filled slot can be "undone" with the
                # corresponding nucleotide action.
                # Backward action a (0..3) is valid if the rightmost filled position
                # contains nucleotide a.
                last_pos = (self.tensor != EMPTY).sum(dim=-1) - 1  # (*batch,)
                valid = last_pos >= 0  # (*batch,)
                # Clamp to avoid out-of-bounds index for fully empty states.
                last_pos_clamped = last_pos.clamp(min=0)
                last_nt = self.tensor.gather(
                    -1, last_pos_clamped.unsqueeze(-1)
                ).squeeze(-1)  # (*batch,)
                # Only set backward mask for valid states (at least one filled pos).
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
            EMPTY, VOCAB_SIZE, (*batch_shape, self.seq_len), device=device
        )
        return self.States(tensor)

    # ------------------------------------------------------------------
    # Forward / backward transitions
    # ------------------------------------------------------------------

    def step(self, states: States, actions: Actions) -> States:
        """Place the chosen nucleotide at the leftmost empty (-1) position."""
        new_tensor = states.tensor.clone()
        action_vals = actions.tensor.squeeze(-1)  # (*batch,)

        # Find the leftmost empty position for each state.
        # is_empty: (*batch, seq_len)
        is_empty = new_tensor == EMPTY
        # leftmost_empty: (*batch,) — index of the first -1, or seq_len if none.
        leftmost_empty = is_empty.long().argmax(dim=-1)

        # Place the nucleotide at the leftmost empty position.
        # Only place for non-exit actions (placement actions < VOCAB_SIZE).
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
        # Find the rightmost filled position.
        is_filled = new_tensor != EMPTY
        # argmax on reversed gives us the leftmost in reversed == rightmost in original.
        rightmost_filled = self.seq_len - 1 - is_filled.flip(-1).long().argmax(dim=-1)
        # Set that position back to EMPTY.
        new_tensor.scatter_(-1, rightmost_filled.unsqueeze(-1), EMPTY)
        return self.States(new_tensor)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def log_reward(self, final_states: DiscreteStates) -> torch.Tensor:
        """log R(x) = -beta * energy(x).

        Lower energy sequences get higher reward, consistent with the nucleobench
        convention that model_fn returns an energy to be minimised.

        Args:
            final_states: terminal states (all positions filled).

        Returns:
            Tensor of shape (*batch_shape,) with log-rewards.
        """
        seqs = _states_to_strings(final_states.tensor, VOCAB)
        energies = self.model_fn(seqs)
        if not isinstance(energies, torch.Tensor):
            energies = torch.tensor(energies, dtype=torch.float32)
        log_r = -self.beta * energies.to(final_states.tensor.device)
        assert log_r.shape == final_states.batch_shape
        return log_r


# ------------------------------------------------------------------
# Helper: decode integer tensor to DNA strings
# ------------------------------------------------------------------


def _states_to_strings(tensor: torch.Tensor, vocab: list[str]) -> list[str]:
    """Convert a batch of integer-encoded states to DNA strings.

    Args:
        tensor: shape (*batch, seq_len) with values in 0..VOCAB_SIZE-1.
        vocab: nucleotide list (e.g. ["A", "C", "G", "T"]).

    Returns:
        List of strings, one per batch element.
    """
    flat = tensor.view(-1, tensor.shape[-1]).cpu().numpy()
    vocab_arr = np.array(vocab)
    return ["".join(vocab_arr[row]) for row in flat]


# ------------------------------------------------------------------
# Build GFlowNet (policy nets + TBGFlowNet)
# ------------------------------------------------------------------


def build_gflownet(
    env: DNASequenceEnv,
    hidden_dim: int = 128,
) -> tuple[TBGFlowNet, Sampler]:
    """Construct a TBGFlowNet with MLP forward and backward policy estimators.

    Uses constant (uniform) backward policy via constant_pb=True in TBGFlowNet.
    The forward policy MLP maps state vectors of length seq_len to logits over
    N_ACTIONS actions.

    Args:
        env: The DNASequenceEnv to train on.
        hidden_dim: Width of MLP hidden layers.

    Returns:
        (gflownet, sampler) ready for training.
    """
    input_dim = env.seq_len

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
    """Sample n sequences from the current policy.

    Args:
        sampler: On-policy sampler wrapping the forward policy estimator.
        env: The DNA sequence environment.
        n: Number of sequences to sample.

    Returns:
        List of n DNA strings, each of length env.seq_len.
    """
    with torch.no_grad():
        trajectories = sampler.sample_trajectories(env=env, n=n, save_logprobs=False)
    terminal = trajectories.terminating_states
    return _states_to_strings(terminal.tensor, VOCAB)
