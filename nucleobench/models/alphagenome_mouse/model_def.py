"""Local AlphaGenome all_folds mouse RNA-seq readouts.

To test with real weights:
```zsh
pytest -v -m alphagenome_mouse integration_tests/alphagenome_mouse_integration_test.py
```
"""

import argparse
import hashlib
from importlib import metadata as package_metadata
from importlib import resources
from pathlib import Path

import numpy as np
import torch

from nucleobench.common import constants
from nucleobench.common import genomic_readout as gr
from nucleobench.models.alphagenome_mouse.bridge import (
    JaxSelectedProfile,
    selected_rna_profile,
)
from nucleobench.optimizations import model_class as mc

READOUTS_ = {
    "es_e14": ("EFO:0007075", "total RNA-seq"),
    "es_bruce4": ("EFO:0005483", "polyA plus RNA-seq"),
}
REPO_ID = "google/alphagenome-all-folds"
REVISION = "a8f293a76ee73d5b57f3bf2ae146510589fcf187"
METADATA_SHA256 = "a22eb7b2e435ed5d34325eb2ffc575679adc1fc6d91958f8c3bf4c7c8a55a929"
# Column names in the upstream RNA_SEQ metadata table. `Assay title` really
# does carry a space and a capital.
ONTOLOGY_COLUMN_ = "ontology_curie"
ASSAY_COLUMN_ = "Assay title"
CODE_SHA256 = {
    "model/dna_model.py": "60e814559844a9f1b8b9663b70f94c776d0a7c5c0f10292bf60ad93a8435ebd0",
    "model/model.py": "199e823b358ec59ff7c81539f1c33bfc1c89fcddbcb69f7fe0302a7571474b9e",
}


class AlphaGenomeMouse(mc.PyTorchDifferentiableModel, mc.TISMModelClass):
    """Predicts mouse ES-cell RNA-seq coverage over a gene's exons."""

    sequence_length = 1_048_576
    output_resolution = 1
    output_origin = 0

    vocab = list(constants.VOCAB)
    vocab_to_idx = {base: i for i, base in enumerate(constants.VOCAB)}
    vocab_array = np.array(list(constants.VOCAB))

    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser()
        group = parser.add_argument_group("AlphaGenome mouse init args")
        group.add_argument(
            "--readout", type=str, required=True, choices=list(READOUTS_)
        )
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
        return {"readout": "es_e14", "strand": "+", "exons": [(0, 1)]}

    def __init__(
        self,
        readout: str,
        strand: str,
        exons: list[tuple[int, int]],
        override_model=None,
        override_weights_local_path: str | None = None,
        override_metadata=None,
        device: str | None = None,
    ):
        if readout not in READOUTS_ or strand not in ("+", "-"):
            raise ValueError("Select E14 or Bruce4 and a gene strand")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.readout, self.strand = readout, strand
        self.exons = gr.union_intervals(exons, 0, self.sequence_length)
        # Static given the exons, so it is built once here rather than on every
        # scored sequence.
        self._coverage_weights = gr.coverage_weights(
            self.sequence_length,
            self.exons,
            origin=0,
            resolution=1,
            device=self.device,
        )
        if override_model is None:
            from alphagenome.models.v1 import dna_model as dna_api
            from alphagenome_research.model import dna_model
            from huggingface_hub import snapshot_download

            package_root = resources.files("alphagenome_research")
            for filename, expected_sha256 in CODE_SHA256.items():
                actual_sha256 = hashlib.sha256(
                    package_root.joinpath(filename).read_bytes()
                ).hexdigest()
                if actual_sha256 != expected_sha256:
                    raise ValueError(f"AlphaGenome upstream code differs: {filename}")
            metadata_bytes = package_root.joinpath(
                "model/metadata/OutputMetadataResponse_ORGANISM_MUS_MUSCULUS.textproto"
            ).read_bytes()
            self.metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
            if self.metadata_sha256 != METADATA_SHA256:
                raise ValueError(
                    "AlphaGenome mouse metadata differs from pinned source"
                )
            snapshot = override_weights_local_path or snapshot_download(
                REPO_ID, revision=REVISION
            )
            self.checkpoint_hashes = {}
            for filename in ("_CHECKPOINT_METADATA", "_METADATA", "manifest.ocdbt"):
                file = Path(snapshot) / filename
                if file.is_file():
                    self.checkpoint_hashes[filename] = hashlib.sha256(
                        file.read_bytes()
                    ).hexdigest()
            organism = dna_api.Organism.MUS_MUSCULUS
            import jax

            accelerators = [
                device
                for device in jax.local_devices()
                if device.platform in ("gpu", "tpu")
            ]
            override_model = dna_model.create(
                snapshot,
                organism_settings={organism: dna_model.OrganismSettings()},
                device=accelerators[0] if accelerators else jax.local_devices()[0],
            )
        self.model = override_model
        if override_metadata is None:
            from alphagenome.models.v1 import dna_model as dna_api
            from alphagenome.models.v1 import dna_output

            organism = dna_api.Organism.MUS_MUSCULUS
            metadata = self.model._metadata[organism].get(dna_output.OutputType.RNA_SEQ)
        else:
            metadata = override_metadata
        self.output_metadata = metadata
        ontology, assay = READOUTS_[readout]
        missing = {ONTOLOGY_COLUMN_, ASSAY_COLUMN_, "strand", "name"} - set(
            metadata.columns
        )
        if missing:
            raise ValueError(
                f"AlphaGenome RNA metadata is missing columns {sorted(missing)}"
            )
        # Positional, because `track_index` indexes the profile array.
        matches = np.flatnonzero(
            (
                (metadata[ONTOLOGY_COLUMN_] == ontology)
                & (metadata[ASSAY_COLUMN_] == assay)
                & (metadata["strand"] == strand)
            ).to_numpy()
        )
        if len(matches) != 1:
            raise ValueError(
                f"AlphaGenome RNA metadata has {len(matches)} matches for "
                f"{ontology} / {assay} / strand {strand}; expected exactly one"
            )
        self.track_index = int(matches[0])
        self.output_names = metadata["name"].tolist()
        self.provenance = {
            "model": "AlphaGenome all_folds mouse",
            "research_version": package_metadata.version("alphagenome_research")
            if override_metadata is None
            else None,
            "checkpoint_revision": REVISION,
            "checkpoint_hashes": getattr(self, "checkpoint_hashes", {}),
            "mouse_metadata_sha256": getattr(self, "metadata_sha256", None),
            "track_index": self.track_index,
            "track_name": self.output_names[self.track_index],
            "strand": strand,
            "exons": self.exons,
            "input_precision": "float32",
            "aggregation": "negative_log1p_union_exon_mean",
            "trunk_rematerialization": True,
            "freeze_trunk_embeddings": False,
        }

    def _jax_profile(self, x):
        return selected_rna_profile(self.model, x, self.track_index)

    def predict_raw_on_tensor(self, x: torch.Tensor) -> torch.Tensor:
        gr.validate_tensor(x, self.sequence_length)
        return JaxSelectedProfile.apply(x, self._jax_profile)

    def string_to_onehot(self, x: list[str]) -> torch.Tensor:
        return gr.one_hot(x, self.sequence_length, self.device)

    def predict_raw_on_strings(self, x: list[str]) -> np.ndarray:
        out = self.predict_raw_on_tensor(self.string_to_onehot(x))
        return out.detach().clone().cpu().numpy()

    def inference_on_tensor(self, x: torch.Tensor) -> torch.Tensor:
        profile = self.predict_raw_on_tensor(x)
        if profile.shape != (x.shape[0], self.sequence_length):
            raise ValueError("Unexpected AlphaGenome RNA profile shape")
        return -torch.log1p(gr.weighted_mean(profile, self._coverage_weights))

    def inference_on_strings(self, x: list[str]) -> np.ndarray:
        if not isinstance(x, (tuple, list)):
            raise ValueError(f"Input needs to be an iterable of strings: {x}")
        out = self.inference_on_tensor(self.string_to_onehot(list(x)))
        return out.detach().clone().cpu().numpy()

    def __call__(self, x: list[str]) -> np.ndarray:
        return self.inference_on_strings(x)
