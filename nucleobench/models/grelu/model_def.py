"""Accessed through gRelu: https://github.com/Genentech/gReLU

Usage follows this tutorial:
https://github.com/Genentech/gReLU/blob/main/docs/tutorials/4_design.ipynb

To test on real data:
```zsh
python -m nucleobench.models.grelu.model_def
```
"""

from collections.abc import Iterable

import grelu.resources
import numpy as np
import pandas as pd
import torch

from nucleobench.models.grelu import constants
from nucleobench.optimizations import model_class as mc


class GReluModel(mc.PyTorchDifferentiableModel, mc.TISMModelClass):
    """General format for gRelu models.

    Specific tasks should inherit from this.
    """

    # List of possible tasks.
    # Set in child models.
    POSSIBLE_TASKS_ = None

    @staticmethod
    def init_parser():
        raise ValueError("Need to be implemented")

    @staticmethod
    def debug_init_args():
        raise ValueError("Need to be implemented")

    def __init__(
        self,
        repo_id: str,
        filename: str,
        expected_sequence_length: int,
        # The vocab MUST be this, since this is what was used in gRelu.
        vocab: list[str] = constants.VOCAB_,
        override_model: torch.nn.Module | None = None,
        device: str = constants.AUTO_DEVICE,
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.device = device
        if self.device == constants.AUTO_DEVICE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if override_model:
            self.model = override_model
        else:
            self.model = grelu.resources.load_model(
                repo_id=self.repo_id, filename=self.filename, device=self.device
            )
            print(f"GRelumodel loaded {self.repo_id}.", flush=True)

        self.tasks = pd.DataFrame(self.model.data_params["tasks"])

        # Consistent vocab is important for interpreting smoothgrad.
        self.vocab = vocab
        self.vocab_to_idx = {nt: i for i, nt in enumerate(vocab)}
        self.vocab_array = np.array(vocab)

        self.has_cuda = torch.cuda.is_available()

        # Check length.
        if "train_seq_len" in self.model.data_params:
            assert self.model.data_params["train_seq_len"] == expected_sequence_length
        else:
            assert "train" in self.model.data_params, self.model.data_params.keys()
            assert (
                self.model.data_params["train"]["seq_len"] == expected_sequence_length
            )

        self.sequence_length = expected_sequence_length

    def inference_on_tensor(
        self,
        x: torch.Tensor,
        return_debug_info: bool = False,
    ) -> torch.Tensor:
        """Run inference on a one-hot tensor."""
        raise ValueError("Implement me.")

    def string_to_onehot(self, x: list[str]) -> torch.Tensor:
        # NOTE: `grelu.sequence.format.strings_to_one_hot(x)` is equivalent to
        # `string_utils.dna2tensor_batch(x, vocab_list=self.vocab)`
        return grelu.sequence.format.strings_to_one_hot(x).to(self.device)

    def inference_on_strings(
        self, x: Iterable[str], return_debug_info: bool = False
    ) -> np.ndarray:
        if not isinstance(x, (tuple, list)):
            raise ValueError(
                f"Input needs to be an iterable of strings, not just string: {x}"
            )
        for s in x:
            if not isinstance(s, str):
                raise ValueError(
                    f"Input needs to be an iterable of strings, instead found: {type(s)}, {s}"
                )

        tensor = self.string_to_onehot(x)
        ret = self.inference_on_tensor(tensor, return_debug_info=return_debug_info)
        if return_debug_info:
            assert len(ret) == 2
            return ret[0].detach().clone().cpu().numpy(), ret[1]
        else:
            return ret.detach().clone().cpu().numpy()

    def __call__(self, x: Iterable[str], return_debug_info: bool = False) -> np.ndarray:
        return self.inference_on_strings(x, return_debug_info=return_debug_info)
