"""
To run:
```zsh
pytest -v -m alphagenome_mouse integration_tests/alphagenome_mouse_integration_test.py
```

Requires access to the gated `all_folds` checkpoint: accept the model terms at
https://huggingface.co/google/alphagenome-all-folds and run `hf auth login`,
or point `ALPHAGENOME_CHECKPOINT_PATH` at a local snapshot. Without one of
those these tests fail rather than skip, so that a scheduled run cannot pass
while testing nothing.

The metadata case only reads the checkpoint's track table, so it is cheap
enough to run in CI. The forward cases each run a full 1 Mb pass (~49 CPU
minutes, ~9 GB peak), so they are opt-in behind `ALPHAGENOME_FORWARD=1`. The
backward pass additionally needs a JAX GPU, behind `ALPHAGENOME_GPU=1`.
"""

import os

import numpy as np
import pytest
import torch

from nucleobench.common.genomic_readout import one_hot
from nucleobench.models.alphagenome_mouse.model_def import (
    ASSAY_COLUMN_,
    ONTOLOGY_COLUMN_,
    AlphaGenomeMouse,
)

SEQUENCE_ = "ACGT" * (1_048_576 // 4)
SEQUENCE_SHA256_ = "29648804bac5219fdd8583062e30250e2c51980a9581ab0df096750b24e32386"
EXONS_ = [(524_288, 524_320)]

READOUTS_ = [
    ("es_e14", "+", "EFO:0007075", "total RNA-seq"),
    ("es_e14", "-", "EFO:0007075", "total RNA-seq"),
    ("es_bruce4", "+", "EFO:0005483", "polyA plus RNA-seq"),
    ("es_bruce4", "-", "EFO:0005483", "polyA plus RNA-seq"),
]


_SKIP_FORWARD = pytest.mark.skipif(
    not os.environ.get("ALPHAGENOME_FORWARD"),
    reason="A 1 Mb forward pass costs ~49 CPU minutes; set ALPHAGENOME_FORWARD=1.",
)


def _has_jax_gpu() -> bool:
    jax = pytest.importorskip("jax")
    return any(device.platform == "gpu" for device in jax.local_devices())


@pytest.fixture(scope="module")
def loaded_model():
    pytest.importorskip("jax")
    pytest.importorskip("alphagenome_research")
    try:
        return AlphaGenomeMouse(
            "es_e14",
            "+",
            EXONS_,
            override_weights_local_path=os.environ.get("ALPHAGENOME_CHECKPOINT_PATH"),
        )
    except Exception as error:
        raise AssertionError(
            "AlphaGenome checkpoint did not load. This needs HF_TOKEN from an "
            "account that accepted the terms at "
            "https://huggingface.co/google/alphagenome-all-folds, or "
            f"ALPHAGENOME_CHECKPOINT_PATH pointing at a local snapshot: {error}"
        ) from error


@pytest.mark.alphagenome_mouse
@pytest.mark.parametrize(("readout", "strand", "ontology", "assay"), READOUTS_)
def test_metadata_resolves_every_readout(
    loaded_model, readout, strand, ontology, assay
):
    """Each readout and strand must name exactly one track in the checkpoint.

    This reads only the track table, so it costs a download rather than a
    forward pass. It is the case that catches upstream renaming a metadata
    column or reshuffling the RNA-seq tracks, which silently breaks every
    readout this adapter offers.
    """
    model = AlphaGenomeMouse(readout, strand, EXONS_, override_model=loaded_model.model)
    rows = model.output_metadata
    assert {ONTOLOGY_COLUMN_, ASSAY_COLUMN_, "strand", "name"} <= set(rows.columns)
    matches = np.flatnonzero(
        (
            (rows[ONTOLOGY_COLUMN_] == ontology)
            & (rows[ASSAY_COLUMN_] == assay)
            & (rows["strand"] == strand)
        ).to_numpy()
    )
    assert len(matches) == 1, f"{ontology}/{assay}/{strand} -> {len(matches)} tracks"
    assert model.track_index == int(matches[0])
    assert rows["name"].iloc[model.track_index] == f"{ontology} {assay}"


@pytest.mark.alphagenome_mouse
@_SKIP_FORWARD
@pytest.mark.parametrize(("readout", "strand", "ontology", "assay"), READOUTS_)
def test_profile_matches_public_predict_sequence(
    loaded_model, readout, strand, ontology, assay
):
    import hashlib

    from alphagenome.models.v1 import dna_model as dna_api
    from alphagenome.models.v1 import dna_output

    assert hashlib.sha256(SEQUENCE_.encode()).hexdigest() == SEQUENCE_SHA256_
    model = AlphaGenomeMouse(readout, strand, EXONS_, override_model=loaded_model.model)
    public = model.model.predict_sequence(
        SEQUENCE_,
        organism=dna_api.Organism.MUS_MUSCULUS,
        requested_outputs=[dna_output.OutputType.RNA_SEQ],
        ontology_terms=[ontology],
    ).rna_seq
    # `predict_sequence` round-trips metadata through a protobuf that keeps the
    # same column names as the checkpoint's own table.
    rows = public.metadata
    match = np.flatnonzero(
        (
            (rows[ONTOLOGY_COLUMN_] == ontology)
            & (rows[ASSAY_COLUMN_] == assay)
            & (rows["strand"] == strand)
        ).to_numpy()
    )
    assert len(match) == 1
    expected = public.values[:, int(match[0])]
    observed = model.predict_raw_on_strings([SEQUENCE_])[0]
    np.testing.assert_allclose(observed, expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        model([SEQUENCE_]),
        [-np.log1p(expected[EXONS_[0][0] : EXONS_[0][1]].mean())],
        rtol=1e-4,
    )


@pytest.mark.alphagenome_mouse
@_SKIP_FORWARD
def test_stochasticity(loaded_model):
    first = loaded_model.predict_raw_on_strings([SEQUENCE_])[0]
    second = loaded_model.predict_raw_on_strings([SEQUENCE_])[0]
    np.testing.assert_allclose(second, first, rtol=0, atol=0)


@pytest.mark.alphagenome_mouse
@_SKIP_FORWARD
def test_batch_matches_single(loaded_model):
    pair = loaded_model.predict_raw_on_strings([SEQUENCE_, SEQUENCE_])
    single = loaded_model.predict_raw_on_strings([SEQUENCE_])
    np.testing.assert_allclose(pair[0], single[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(pair[1], single[0], rtol=1e-4, atol=1e-4)


@pytest.mark.alphagenome_mouse
@pytest.mark.skipif(
    not os.environ.get("ALPHAGENOME_GPU"),
    reason="A full-length 1 Mb backward pass needs a JAX GPU; set ALPHAGENOME_GPU=1.",
)
def test_full_length_input_gradient(loaded_model):
    assert _has_jax_gpu(), "ALPHAGENOME_GPU is set but no JAX GPU device is visible"
    x = one_hot([SEQUENCE_], loaded_model.sequence_length).requires_grad_(True)
    loaded_model.inference_on_tensor(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
