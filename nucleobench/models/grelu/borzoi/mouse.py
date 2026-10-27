"""Mouse Borzoi replicate 0 exon-coverage objective.

Same architecture and loader as the human Borzoi in `model_def.py`, but with the
mouse head (2,608 tracks) and a gene-centric readout: the score is the negative
`log1p` of overlap-weighted mean coverage across a gene's exons.

To test with real weights:
```zsh
pytest -v -m borzoi_mouse integration_tests/borzoi_mouse_integration_test.py
```
"""

import argparse
from importlib import metadata as package_metadata

import numpy as np
import torch

from nucleobench.common import genomic_readout as gr
from nucleobench.models.grelu import constants as grelu_constants
from nucleobench.models.grelu import model_def as grelu_md
from nucleobench.models.grelu.borzoi import constants

REPO_ID = "Genentech/borzoi-model"
FILENAME = "mouse_rep0.ckpt"
# Pinned so that a silently re-uploaded checkpoint is caught rather than used.
SHA256 = "9c8aeaa8633d1405913afe95077dd78c5a269100ee2e61c47518f1fa4b102b57"

N_MOUSE_TASKS_ = 2608
N_SPATIAL_BINS_ = 6144
# Borzoi predicts the center 196,608 bp of its input in 32-bp bins.
OUTPUT_RESOLUTION_ = 32
OUTPUT_ORIGIN_ = 163_840

# ENCODE RNA-seq tracks for the two mouse ES lines. E14 is stranded, so the
# index depends on the gene's strand; Bruce4 is unstranded.
TRACKS_ = {
    "es_e14": {"+": (2287, "ENCFF968GGL+"), "-": (2288, "ENCFF968GGL-")},
    "es_bruce4": {"+": (2237, "ENCFF427XNM"), "-": (2237, "ENCFF427XNM")},
}


class BorzoiMouse(grelu_md.GReluModel):
    """Predicts mouse ES-cell RNA-seq coverage over a gene's exons."""

    sequence_length = constants.BORZOI_TRAIN_LEN_
    output_resolution = OUTPUT_RESOLUTION_
    output_origin = OUTPUT_ORIGIN_

    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser()
        group = parser.add_argument_group("Mouse Borzoi init args")
        group.add_argument("--readout", type=str, required=True, choices=list(TRACKS_))
        group.add_argument("--strand", type=str, required=True, choices=["+", "-"])
        group.add_argument(
            "--exons",
            type=gr.parse_exons,
            required=True,
            help="Comma-separated `start:end` offsets into the input sequence.",
        )
        group.add_argument("--override_weights_local_path", type=str, default=None)
        return parser

    @staticmethod
    def debug_init_args():
        return {
            "readout": "es_e14",
            "strand": "+",
            "exons": [(OUTPUT_ORIGIN_, OUTPUT_ORIGIN_ + OUTPUT_RESOLUTION_)],
            "run_sanity_checks": False,
        }

    def __init__(
        self,
        readout: str,
        strand: str,
        exons: list[tuple[int, int]],
        override_model: torch.nn.Module | None = None,
        override_weights_local_path: str | None = None,
        device: str = grelu_constants.AUTO_DEVICE,
        run_sanity_checks: bool = True,
    ):
        if readout not in TRACKS_ or strand not in ("+", "-"):
            raise ValueError("Select a named readout and gene strand")
        if device == grelu_constants.AUTO_DEVICE:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.readout, self.strand = readout, strand
        self.track_index, self.track_name = TRACKS_[readout][strand]
        # Validate coordinates before downloading a 700 MB checkpoint.
        self.exons = gr.union_intervals(
            exons,
            OUTPUT_ORIGIN_,
            OUTPUT_ORIGIN_ + N_SPATIAL_BINS_ * OUTPUT_RESOLUTION_,
        )

        self.weights_sha256: str | None = None
        if override_model is None:
            # `GReluModel` would call `grelu.resources.load_model`, which skips
            # the integrity check, so resolve and verify the file ourselves and
            # hand the loaded model to the base class.
            import grelu.resources
            from grelu.lightning import LightningModel

            path = override_weights_local_path or grelu.resources.download_model(
                repo_id=REPO_ID, filename=FILENAME
            )
            self.weights_sha256 = gr.file_sha256(path)
            if self.weights_sha256 != SHA256:
                raise ValueError(
                    f"Mouse Borzoi checkpoint SHA256 mismatch: {self.weights_sha256}"
                )
            override_model = LightningModel.load_from_checkpoint(
                path, map_location=device
            )

        super().__init__(
            repo_id=REPO_ID,
            filename=FILENAME,
            expected_sequence_length=constants.BORZOI_TRAIN_LEN_,
            override_model=override_model,
            device=device,
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self._check_tracks()
        # Static given the exons and the output grid, so it is built once here
        # rather than on every scored sequence.
        self._coverage_weights = gr.coverage_weights(
            N_SPATIAL_BINS_,
            self.exons,
            origin=OUTPUT_ORIGIN_,
            resolution=OUTPUT_RESOLUTION_,
            device=self.device,
        )
        self.output_names = self.tasks["name"].tolist()
        self.output_metadata = self.tasks
        self.provenance = {
            "model": "Borzoi mouse replicate 0",
            "grelu_version": package_metadata.version("grelu"),
            "checkpoint_repo_id": REPO_ID,
            "weights_sha256": self.weights_sha256,
            "track_index": self.track_index,
            "track_name": self.track_name,
            "strand": self.strand,
            "exons": self.exons,
            "input_precision": "float32",
            "aggregation": "negative_log1p_union_exon_mean",
        }

        if run_sanity_checks:
            out = self.model(self.string_to_onehot(["A" * self.sequence_length]))
            assert out.shape == (1, N_MOUSE_TASKS_, N_SPATIAL_BINS_), out.shape

    def _check_tracks(self) -> None:
        """Fail loudly if the checkpoint's target table is not what we indexed."""
        tasks = self.tasks  # Set by `GReluModel` from `model.data_params["tasks"]`.
        if len(tasks) != N_MOUSE_TASKS_:
            raise ValueError(
                f"Mouse Borzoi expects {N_MOUSE_TASKS_} tracks, found {len(tasks)}"
            )
        if tasks["name"].iloc[self.track_index] != self.track_name:
            raise ValueError(
                "Mouse Borzoi target table does not match expected track indices"
            )
        # Scores are on the checkpoint's training scale, not physical read counts.
        if "sum_stat" in tasks.columns and (
            tasks["sum_stat"].iloc[self.track_index] != "sum_sqrt"
            or float(tasks["scale"].iloc[self.track_index]) != 0.3
        ):
            raise ValueError("Mouse Borzoi RNA target transformation differs")

    def string_to_onehot(self, x: list[str]) -> torch.Tensor:
        # Overrides the gReLU encoder, which accepts any length and rejects
        # lowercase. Exon offsets are only meaningful at the exact input length.
        return gr.one_hot(x, self.sequence_length, self.device)

    def predict_raw_on_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Full (batch, 2608, 6144) track-by-bin prediction."""
        gr.validate_tensor(x, self.sequence_length)
        out = self.model(x.to(device=self.device, dtype=torch.float32))
        if out.shape != (x.shape[0], N_MOUSE_TASKS_, N_SPATIAL_BINS_):
            raise ValueError(
                f"Unexpected mouse Borzoi output shape: {tuple(out.shape)}"
            )
        return out

    def predict_raw_on_strings(self, x: list[str]) -> np.ndarray:
        out = self.predict_raw_on_tensor(self.string_to_onehot(x))
        return out.detach().clone().cpu().numpy()

    def inference_on_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Negative log1p of exon-mean coverage; lower means more expression."""
        profile = self.predict_raw_on_tensor(x)[:, self.track_index, :]
        coverage = gr.weighted_mean(profile, self._coverage_weights)
        # Always return something that should be minimized.
        return -torch.log1p(coverage)
