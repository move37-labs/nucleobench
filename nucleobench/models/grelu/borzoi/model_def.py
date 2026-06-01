"""RNA-seq coverage from DNA: https://www.nature.com/articles/s41588-024-02053-6

Accessed through gRelu: https://github.com/Genentech/gReLU

Usage follows this tutorial:
https://github.com/Genentech/gReLU/blob/main/docs/tutorials/4_design.ipynb

To test on real data:
```zsh
python -m nucleobench.models.grelu.borzoi.model_def
```
"""

import argparse
import os

import torch

from nucleobench.models.grelu import model_def as grelu_md
from nucleobench.models.grelu.borzoi import constants
from nucleobench.models.grelu.enformer import constants as enformer_constants


class Borzoi(grelu_md.GReluModel):
    """Predicts RNA-seq coverage from DNA.
    https://www.nature.com/articles/s41588-024-02053-6
    """

    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser()
        group = parser.add_argument_group("Borzoi init args")
        group.add_argument(
            "--aggregation_type", type=str, required=True, choices=["muscle_not_liver"]
        )

        return parser

    @staticmethod
    def debug_init_args():
        return {
            "aggregation_type": "muscle_not_liver",
            "run_sanity_checks": False,
        }

    @staticmethod
    def write_dummy_sequence_to_file(
        filepath: str, sequence_length: int = constants.BORZOI_TRAIN_LEN_
    ) -> None:
        """Write a dummy sequence of specified length to a text file.

        Args:
            filepath: Path to the output file
            sequence_length: Length of the sequence to generate (default: 524_288)
        """
        # Generate a dummy sequence (to be determined later, using 'A' as placeholder)
        dummy_sequence = "A" * sequence_length

        # Ensure directory exists
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )

        with open(filepath, "w") as f:
            f.write(dummy_sequence)

    @staticmethod
    def inject_middle_sequence(
        base_sequence: str,
        middle_sequence: str,
        total_length: int = constants.BORZOI_TRAIN_LEN_,
        middle_length: int = enformer_constants.ENFORMER_TRAIN_LEN_,
    ) -> str:
        """Inject a middle sequence into a base sequence."""
        assert len(base_sequence) == total_length, (
            f"Base sequence length {len(base_sequence)} != expected {total_length}"
        )
        assert len(middle_sequence) == middle_length, (
            f"Middle sequence length {len(middle_sequence)} != expected {middle_length}"
        )

        # Calculate padding on each side
        padding_length = (total_length - middle_length) // 2

        # Construct: [left_pad] + [middle_sequence] + [right_pad]
        # Use the original sequence's left and right parts as padding
        left_pad = base_sequence[:padding_length]
        right_pad = base_sequence[padding_length + middle_length :]

        modified_sequence = left_pad + middle_sequence + right_pad

        assert len(modified_sequence) == total_length, (
            f"Modified sequence length {len(modified_sequence)} != expected {total_length}"
        )

        return modified_sequence

    @staticmethod
    def read_and_inject_middle_sequence(
        filepath: str,
        middle_sequence: str,
        total_length: int = constants.BORZOI_TRAIN_LEN_,
        middle_length: int = enformer_constants.ENFORMER_TRAIN_LEN_,
    ) -> str:
        """Read a sequence from a file and inject a middle sequence.

        Args:
            filepath: Path to the input sequence file
            middle_sequence: Sequence to inject in the middle (should be middle_length long)
            total_length: Total length of the output sequence (default: 524_288)
            middle_length: Length of the middle sequence to inject (default: 196_608)

        Returns:
            Modified sequence with middle_sequence injected in the middle
        """
        with open(filepath) as f:
            base_sequence = f.read().strip()

        return Borzoi.inject_middle_sequence(
            base_sequence=base_sequence,
            middle_sequence=middle_sequence,
            total_length=total_length,
            middle_length=middle_length,
        )

    @staticmethod
    def enformer_spatial_bins(total_bins: int = 6144) -> list[int]:
        """Return the spatial bins that correspond to the enformer model."""
        borzoi_len = constants.BORZOI_TRAIN_LEN_
        enformer_len = enformer_constants.ENFORMER_TRAIN_LEN_

        middle_bins = int(round((enformer_len / borzoi_len) * total_bins))
        pad_bins = (total_bins - middle_bins) // 2

        start_bin = pad_bins
        end_bin = pad_bins + middle_bins
        return list(range(start_bin, end_bin))

    # List of possible tasks.
    # Set in child models.
    POSSIBLE_TASKS_ = constants.BORZOI_TASKS_

    def __init__(
        self,
        aggregation_type: str,
        spatial_bins_to_aggregate: list[int] | None = None,
        override_model: torch.nn.Module | None = None,
        override_aggregation=None,
        run_sanity_checks: bool = True,
    ):
        super().__init__(
            repo_id=constants.BORZOI_REPO_ID,
            filename=constants.BORZOI_FILENAME,
            override_model=override_model,
            expected_sequence_length=constants.BORZOI_TRAIN_LEN_,
        )
        self.model.eval()

        if aggregation_type not in ["muscle_not_liver"]:
            raise ValueError(f"Unknown aggregation type: {aggregation_type}")

        if override_aggregation is None:
            positive_idxs, negative_idxs = constants.idxs_by_name(aggregation_type)

            def _aggregation(model_out: torch.Tensor) -> torch.Tensor:
                assert model_out.ndim == 3
                assert model_out.shape[1] == len(constants.BORZOI_TASKS_)

                if spatial_bins_to_aggregate is not None:
                    model_out = model_out[:, :, spatial_bins_to_aggregate]

                ret = torch.sum(model_out[:, positive_idxs], axis=(1, 2)) - torch.sum(
                    model_out[:, negative_idxs], axis=(1, 2)
                )
                assert ret.ndim == 1
                return ret

            self.aggregation = _aggregation
        else:
            self.aggregation = override_aggregation

        # Sanity check inference.
        if run_sanity_checks:
            ret = self.model(self.string_to_onehot(["A" * self.sequence_length]))
            assert ret.shape == (1, 7611, 6144), ret.shape

            ret = self.inference_on_strings(["A" * self.sequence_length])
            assert ret.ndim == 1

    def inference_on_tensor(
        self,
        x: torch.Tensor,
        return_debug_info: bool = False,
    ) -> torch.Tensor:
        """Run inference on a one-hot tensor."""
        del return_debug_info
        assert x.ndim == 3  # Batched.
        assert x.shape[1] == 4
        # assert x.shape[2] == self.sequence_length, x.shape

        m_out = self.model(x)
        assert m_out.ndim == 3
        assert m_out.shape[1] == len(constants.BORZOI_TASKS_), m_out.shape

        ret = self.aggregation(m_out)
        assert ret.ndim == 1, ret.shape

        # Always return something that should be minimized, so flip the sign.
        ret *= -1

        return ret


if __name__ == "__main__":
    # Test with a real model.
    import time

    import tqdm

    print("Starting muscle_not_liver...")
    m = Borzoi(aggregation_type="muscle_not_liver")
    ntimes = 1
    s_time = time.time()
    for _ in tqdm.trange(ntimes):
        o = m.model(m.string_to_onehot(["A" * 524_288]))
        print(f"Output shape: {o.shape}")
    e_time = time.time()
    print(
        f"Finished in {e_time - s_time} seconds: {(e_time - s_time) / ntimes} s / iter"
    )
