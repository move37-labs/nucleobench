"""RNA-seq coverage from DNA: https://www.nature.com/articles/s41588-024-02053-6

Accessed through gRelu: https://github.com/Genentech/gReLU

Usage follows this tutorial:
https://github.com/Genentech/gReLU/blob/main/docs/tutorials/4_design.ipynb

Mouse replicate 0 (`mouse_rep0.ckpt`), 2,608 tracks. Scores a gene rather than
a tissue contrast: the negative `log1p` of overlap-weighted mean coverage
across the gene's exons. The human variant is in `../human/model_def.py`.

To test on real data:
```zsh
python -m nucleobench.models.grelu.borzoi.mouse.model_def
```

To run the full integration suite:
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
from nucleobench.models.grelu.borzoi.mouse import constants as mouse_constants

# Re-exported so callers and tests can use the short names.
REPO_ID = constants.BORZOI_REPO_ID
FILENAME = mouse_constants.BORZOI_MOUSE_FILENAME
SHA256 = mouse_constants.BORZOI_MOUSE_SHA256
N_MOUSE_TASKS_ = mouse_constants.BORZOI_MOUSE_N_TASKS_
N_SPATIAL_BINS_ = mouse_constants.BORZOI_MOUSE_N_SPATIAL_BINS_
OUTPUT_RESOLUTION_ = mouse_constants.BORZOI_MOUSE_OUTPUT_RESOLUTION_
OUTPUT_ORIGIN_ = mouse_constants.BORZOI_MOUSE_OUTPUT_ORIGIN_
TRACKS_ = mouse_constants.BORZOI_MOUSE_TRACKS_


class BorzoiMouse(grelu_md.GReluModel):
    """Predicts mouse ES-cell RNA-seq coverage over a gene's exons."""

    # List of possible tasks.
    POSSIBLE_TASKS_ = mouse_constants.BORZOI_MOUSE_TASKS_

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
        self.device = device
        self.exons = exons

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

    @property
    def exons(self) -> list[tuple[int, int]]:
        return self._exons

    @exons.setter
    def exons(self, intervals) -> None:
        """Validate exons and rebuild the coverage weights they determine.

        The weights depend only on the exons and the output grid, so they are
        cached rather than rebuilt per forward pass. Going through a setter
        keeps that cache from going stale when a caller rescores a different
        gene with the same loaded model.
        """
        self._exons = gr.union_intervals(
            intervals,
            OUTPUT_ORIGIN_,
            OUTPUT_ORIGIN_ + N_SPATIAL_BINS_ * OUTPUT_RESOLUTION_,
        )
        self._coverage_weights = gr.coverage_weights(
            N_SPATIAL_BINS_,
            self._exons,
            origin=OUTPUT_ORIGIN_,
            resolution=OUTPUT_RESOLUTION_,
            device=self.device,
        )

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


if __name__ == "__main__":
    # Test with a real model.
    import time

    import tqdm

    print("Starting es_e14...")
    m = BorzoiMouse(
        readout="es_e14",
        strand="+",
        exons=[(OUTPUT_ORIGIN_, OUTPUT_ORIGIN_ + OUTPUT_RESOLUTION_)],
        run_sanity_checks=False,
    )
    ntimes = 1
    s_time = time.time()
    for _ in tqdm.trange(ntimes):
        o = m.model(m.string_to_onehot(["A" * 524_288]))
        print(f"Output shape: {o.shape}")
    e_time = time.time()
    print(
        f"Finished in {e_time - s_time} seconds: {(e_time - s_time) / ntimes} s / iter"
    )
