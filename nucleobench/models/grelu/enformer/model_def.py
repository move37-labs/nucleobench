"""Gene expression from DNA: https://www.nature.com/articles/s41592-021-01252-x

Accessed through gRelu: https://github.com/Genentech/gReLU

Usage follows this tutorial:
https://github.com/Genentech/gReLU/blob/main/docs/tutorials/4_design.ipynb

To test on real data:
```zsh
python -m nucleobench.models.grelu.enformer.model_def
```
"""

import argparse

import numpy as np
import torch

from nucleobench.models.grelu import model_def as grelu_md
from nucleobench.models.grelu.enformer import constants


class Enformer(grelu_md.GReluModel):
    """Predicts gene expression from DNA.
    https://www.nature.com/articles/s41592-021-01252-x
    """

    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser()
        group = parser.add_argument_group("Enformer init args")
        group.add_argument(
            "--aggregation_type",
            type=str,
            required=True,
            choices=["muscle_not_liver", "k562_dnase"],
        )
        return parser

    @staticmethod
    def debug_init_args():
        return {
            "aggregation_type": "muscle_not_liver",
            "run_sanity_checks": False,
        }

    # List of possible tasks.
    # Set in child models.
    POSSIBLE_TASKS_ = constants.ENFORMER_TASKS_

    def __init__(
        self,
        aggregation_type: str,
        spatial_bins_to_aggregate: list[int] | None = None,
        override_model: torch.nn.Module | None = None,
        override_aggregation=None,
        run_sanity_checks: bool = True,
    ):
        """Init.

        Aggregation is complicated. We enumerate the options as strings, and use premade aggregation
        functions for well thought-out options, instead of asking the user to provide them, so the
        aggregation can be easily tracked and tested.

        Args:
            aggregation_type: One of "muscle_not_liver" or "k562_dnase".
            spatial_bins_to_aggregate: If set, restrict spatial bins before aggregating.
            override_model: Swap in a fake model (for testing).
            override_aggregation: Bypass the named aggregation entirely (for testing or
                custom callers). When set, aggregation_type is ignored.
            run_sanity_checks: Run a small forward pass to validate the model on init.
        """
        super().__init__(
            repo_id=constants.ENFORMER_REPO_ID,
            filename=constants.ENFORMER_FILENAME,
            override_model=override_model,
            expected_sequence_length=constants.ENFORMER_TRAIN_LEN_,
        )
        self.model.eval()

        if aggregation_type not in ["muscle_not_liver", "k562_dnase"]:
            raise ValueError(f"Unknown aggregation type: {aggregation_type}")

        # Create aggregation method.
        if override_aggregation is None:
            if aggregation_type == "muscle_not_liver":
                positive_idxs, negative_idxs = constants.idxs_by_name(aggregation_type)

                def _aggregation(model_out: torch.Tensor) -> torch.Tensor:
                    assert model_out.ndim == 3
                    assert model_out.shape[1] == len(constants.ENFORMER_TASKS_)

                    # If spatial_bins_to_aggregate is specified, use only those bins.
                    if spatial_bins_to_aggregate is not None:
                        model_out = model_out[:, :, spatial_bins_to_aggregate]

                    ret = torch.sum(model_out[:, positive_idxs], dim=(1, 2)) - torch.sum(
                        model_out[:, negative_idxs], dim=(1, 2)
                    )
                    assert ret.ndim == 1
                    return ret

                self.aggregation = _aggregation

            elif aggregation_type == "k562_dnase":
                track_indices = constants.k562_dnase_track_indices()

                def _aggregation_k562_dnase(model_out: torch.Tensor) -> torch.Tensor:
                    assert model_out.ndim == 3
                    assert model_out.shape[1] == len(constants.ENFORMER_TASKS_)
                    out = model_out[:, track_indices]
                    if spatial_bins_to_aggregate is not None:
                        out = out[:, :, spatial_bins_to_aggregate]
                    ret = out.sum(dim=(1, 2))
                    assert ret.ndim == 1
                    return ret

                self.aggregation = _aggregation_k562_dnase

        else:
            self.aggregation = override_aggregation

        # Sanity check inference.
        if run_sanity_checks:
            ret = self.model(self.string_to_onehot(["A" * self.sequence_length]))
            assert ret.shape == (1, 5313, 896), ret.shape
            # Apparently
            ret = self.model(self.string_to_onehot(["A" * 82_000]))
            assert ret.shape == (1, 5313, 1), ret.shape

            ret = self.inference_on_strings(["A" * self.sequence_length])
            assert isinstance(ret, np.ndarray)
            assert ret.ndim == 1

    def inference_on_tensor(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Run inference on a one-hot tensor."""
        assert x.ndim == 3  # Batched.
        assert x.shape[1] == 4
        # assert x.shape[2] == self.sequence_length, x.shape

        m_out = self.model(x)
        assert m_out.ndim == 3
        assert m_out.shape[1] == len(constants.ENFORMER_TASKS_), m_out.shape

        ret = self.aggregation(m_out)
        assert ret.ndim == 1, ret.shape

        # Always return something that should be minimized, so flip the sign.
        ret *= -1

        return ret


if __name__ == "__main__":
    # Test with a real model.
    import time

    import tqdm

    print("Starting Enformer...")
    m = Enformer(aggregation_type="muscle_not_liver")
    ntimes = 10
    s_time = time.time()
    # Runs at roughly 15s / iteration, on my macbook.
    for _ in tqdm.trange(ntimes):
        ret = m(["A" * 196_608])
    e_time = time.time()
    print(
        f"Finished in {e_time - s_time} seconds: {(e_time - s_time) / ntimes} s / iter"
    )
