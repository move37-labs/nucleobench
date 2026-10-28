"""CREsted DeepHumanBrain normalized peak accessibility objective.

Predicts 76 normalized accessibility values -- one per brain cell type -- each
summarizing the central 1,000 bp of a 2,114 bp input. The outputs are not
position-resolved profiles.

To test with real weights:
```zsh
KERAS_BACKEND=torch pytest -v -m deephumanbrain \
    integration_tests/deephumanbrain_integration_test.py
```
"""

import argparse
import hashlib
import os
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from nucleobench.common import constants
from nucleobench.common import genomic_readout as gr
from nucleobench.optimizations import model_class as mc

N_OUTPUTS_ = 76
SEQUENCE_LENGTH_ = 2114
# The 76 outputs summarize the central 1,000 bp of the input.
OUTPUT_ORIGIN_ = 557

ARCHIVE_SHA256 = "8f1e3d71208a587f64b8d20d24851bda9b0779dc6dca623e9287d1506a357c06"
OUTPUT_NAMES_SHA256 = "c90008790a163423cf3d178db61156dfcf1c57c938826cca799ca8ba4ee18530"


class DeepHumanBrain(mc.PyTorchDifferentiableModel, mc.TISMModelClass):
    """Predicts brain cell-type accessibility from DNA, via CREsted."""

    sequence_length = SEQUENCE_LENGTH_
    output_resolution = None  # A single value per output, not a profile.
    output_origin = OUTPUT_ORIGIN_

    vocab = list(constants.VOCAB)
    vocab_to_idx = {base: i for i, base in enumerate(constants.VOCAB)}
    vocab_array = np.array(list(constants.VOCAB))

    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser()
        group = parser.add_argument_group("DeepHumanBrain init args")
        group.add_argument("--output_names", type=str, nargs="+", required=True)
        group.add_argument("--override_weights_local_path", type=str, default=None)
        group.add_argument("--override_output_names", type=str, nargs="+", default=None)
        return parser

    @staticmethod
    def debug_init_args():
        return {"output_names": ["AMY"]}

    def __init__(
        self,
        output_names: list[str],
        override_model=None,
        override_weights_local_path: str | None = None,
        override_output_names: list[str] | None = None,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path = None
        if override_model is None:
            # CREsted is Keras-backend-agnostic, but only the PyTorch backend
            # gives us differentiable tensors, and the backend is global state
            # that cannot be changed once Keras has initialized.
            if (
                "keras" in sys.modules
                and sys.modules["keras"].backend.backend() != "torch"
            ):
                raise RuntimeError(
                    "Keras was initialized before selecting its PyTorch backend"
                )
            os.environ["KERAS_BACKEND"] = "torch"
            import crested

            if override_weights_local_path:
                if override_output_names is None:
                    raise ValueError("Local CREsted weights require exact output names")
                path, labels = override_weights_local_path, override_output_names
            else:
                path, labels = crested.get_model("DeepHumanBrain")
            override_model = crested.utils.load_model(path)
        else:
            if override_output_names is None:
                raise ValueError("Injected CREsted model requires exact output names")
            labels = override_output_names

        self.output_names = list(labels)
        self.output_metadata = self.output_names
        self.metadata_sha256 = hashlib.sha256(
            "\n".join(self.output_names).encode()
        ).hexdigest()
        if (
            path is not None
            and override_weights_local_path is None
            and self.metadata_sha256 != OUTPUT_NAMES_SHA256
        ):
            raise ValueError(
                "DeepHumanBrain output names differ from the pinned metadata"
            )
        if (
            len(self.output_names) != N_OUTPUTS_
            or len(set(self.output_names)) != N_OUTPUTS_
        ):
            raise ValueError(f"DeepHumanBrain requires {N_OUTPUTS_} unique labels")
        if not output_names or any(
            label not in self.output_names for label in output_names
        ):
            raise ValueError("Select exact CREsted output labels")
        self.selected_indices = [
            self.output_names.index(label) for label in output_names
        ]

        self.model = override_model
        self.weights_sha256: str | None = None
        if path is not None and Path(path).is_file():
            self.weights_sha256 = gr.file_sha256(path)

        # Keras picks its own device when it builds the model, which need not
        # be the one the caller asked for -- it selects an accelerator even
        # when `device` is "cpu". Moving it keeps `self.device` honest, so the
        # inputs cast in `predict_raw_on_tensor` land on the same device as
        # the weights.
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "trainable"):
            self.model.trainable = False
        if hasattr(self.model, "eval"):
            self.model.eval()
        if hasattr(self.model, "parameters"):
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

        self.provenance = {
            "model": "DeepHumanBrain",
            "crested_version": metadata.version("crested")
            if path is not None
            else None,
            "archive_sha256": ARCHIVE_SHA256 if path is not None else None,
            "weights_sha256": self.weights_sha256,
            "output_names_sha256": self.metadata_sha256,
            "selected_output_names": list(output_names),
            "input_precision": "float32",
            "aggregation": "negative_mean_normalized_accessibility",
        }

    def string_to_onehot(self, x: list[str]) -> torch.Tensor:
        return gr.one_hot(x, self.sequence_length, self.device)

    def predict_raw_on_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """All 76 cell-type accessibility values, shaped (batch, 76)."""
        gr.validate_tensor(x, self.sequence_length)
        # CREsted models take (batch, length, 4).
        out = self.model(
            x.to(device=self.device, dtype=torch.float32).transpose(1, 2),
            training=False,
        )
        if isinstance(out, (list, tuple)):
            out = out[0]
        if not isinstance(out, torch.Tensor) or out.shape != (x.shape[0], N_OUTPUTS_):
            raise ValueError(
                f"Expected differentiable DeepHumanBrain output (batch, {N_OUTPUTS_})"
            )
        return out

    def predict_raw_on_strings(self, x: list[str]) -> np.ndarray:
        out = self.predict_raw_on_tensor(self.string_to_onehot(x))
        return out.detach().clone().cpu().numpy()

    def inference_on_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Negative mean accessibility; lower means more accessible."""
        return -self.predict_raw_on_tensor(x)[:, self.selected_indices].mean(dim=1)

    def inference_on_strings(self, x: list[str]) -> np.ndarray:
        if not isinstance(x, (tuple, list)):
            raise ValueError(f"Input needs to be an iterable of strings: {x}")
        out = self.inference_on_tensor(self.string_to_onehot(list(x)))
        return out.detach().clone().cpu().numpy()

    def __call__(self, x: list[str]) -> np.ndarray:
        return self.inference_on_strings(x)
