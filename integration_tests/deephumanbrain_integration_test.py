"""
To run:
```zsh
KERAS_BACKEND=torch pytest -v -m deephumanbrain \
    integration_tests/deephumanbrain_integration_test.py
```

Two kinds of check. The fixture test compares the PyTorch-backend CREsted
model against TensorFlow-generated reference predictions for all 76 outputs,
which establishes that the two Keras backends agree numerically. The locus
tests check that the predictions are biologically sensible on real hg38
sequence, which numerical agreement alone does not show.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import requests
import torch

import nucleobench.models.deephumanbrain
from nucleobench.common.genomic_readout import one_hot
from nucleobench.models.deephumanbrain.model_def import DeepHumanBrain

FIXTURE_ = (
    Path(nucleobench.models.deephumanbrain.__file__).parent
    / "testdata/deephumanbrain_tensorflow.json"
)

_L = 2114
# hg38 1-based TSS, ncbiRefSeqCurated. Windows use the same centering as the
# ChromBPNet integration test, which also scores 2114 bp.
_SNAP25 = ("chr20", 10_218_829)  # Pan-neuronal, + strand (NM_003081.5).
_GFAP = ("chr17", 44_915_500)  # Astrocyte-specific, - strand (NM_002055.5).
# Same gene-desert locus as the Enformer / Borzoi / ChromBPNet tests.
_GENE_DESERT = ("chr8", 127_150_000)

_UCSC_SEQUENCE_URL = (
    "https://api.genome.ucsc.edu/getData/sequence"
    "?genome=hg38;chrom={chrom};start={start};end={end}"
)


def _fetch_hg38(chrom: str, center: int, length: int = _L) -> str:
    """Fetch a centered hg38 window from the UCSC REST API."""
    start = center - (length // 2)
    end = center + ((length - 1) // 2)
    url = _UCSC_SEQUENCE_URL.format(chrom=chrom, start=start - 1, end=end)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    sequence = response.json()["dna"].upper()
    if len(sequence) != length:
        raise ValueError(f"Expected {length} bp, got {len(sequence)}")
    return sequence


@pytest.fixture(scope="module")
def locus_predictions():
    """All 76 outputs for two brain promoters and a gene desert."""
    pytest.importorskip("crested")
    model = DeepHumanBrain(["AMY"])
    loci = {"SNAP25": _SNAP25, "GFAP": _GFAP, "GENE_DESERT": _GENE_DESERT}
    sequences = [_fetch_hg38(*coords) for coords in loci.values()]
    predictions = model.predict_raw_on_strings(sequences)
    labels = model.output_names
    return {
        "by_locus": dict(zip(loci, predictions)),
        "astrocyte": [labels.index(n) for n in labels if n.startswith("ASCT")],
        "neuronal": [labels.index(n) for n in labels if n.startswith("ITL23")],
    }


@pytest.mark.deephumanbrain
def test_real_weights_live_on_the_requested_device():
    """Keras builds the real model on an accelerator even when asked for CPU."""
    pytest.importorskip("crested")
    model = DeepHumanBrain(["AMY"], device="cpu")
    devices = {str(parameter.device) for parameter in model.model.parameters()}
    assert devices == {"cpu"}, devices


@pytest.mark.deephumanbrain
def test_neuronal_promoter_more_accessible_than_gene_desert(locus_predictions):
    """A pan-neuronal promoter should stand out against unannotated sequence."""
    by_locus = locus_predictions["by_locus"]
    snap25 = by_locus["SNAP25"].mean()
    desert = by_locus["GENE_DESERT"].mean()
    # Note: These thresholds may need to be tuned.
    assert snap25 > 0.2, "SNAP25 should be broadly accessible across brain types"
    assert desert < 0.1, "A gene desert should be near-closed"
    assert snap25 > 3 * desert


@pytest.mark.deephumanbrain
def test_astrocyte_promoter_is_cell_type_specific(locus_predictions):
    """GFAP should open in astrocytes specifically, not in neurons.

    This is the check that mean accessibility cannot make: GFAP's average over
    all 76 outputs is close to the gene desert's, because only the three
    astrocyte types are open. The signal is in the contrast.
    """
    astro = locus_predictions["astrocyte"]
    neuro = locus_predictions["neuronal"]
    gfap = locus_predictions["by_locus"]["GFAP"]
    desert = locus_predictions["by_locus"]["GENE_DESERT"]
    # Note: These thresholds may need to be tuned.
    assert gfap[astro].mean() > 4 * gfap[neuro].mean(), (
        "GFAP should be far more accessible in astrocytes than in neurons"
    )
    # The desert has no cell-type preference, so the contrast is specific to
    # GFAP rather than an artifact of how the two label groups differ.
    ratio = desert[astro].mean() / desert[neuro].mean()
    assert 0.5 < ratio < 2.0, f"Gene desert should not favor a cell type: {ratio}"


@pytest.mark.deephumanbrain
def test_tensorflow_fixture_public_api_and_input_gradient():
    crested = pytest.importorskip("crested")
    sequences = ["ACGT" * 528 + "AC", "A" * 2114, "C" * 2114]
    fixture = json.loads(FIXTURE_.read_text())
    assert [
        hashlib.sha256(sequence.encode()).hexdigest() for sequence in sequences
    ] == [row["sha256"] for row in fixture["sequences"]]
    model = DeepHumanBrain(["AMY"])
    assert model.metadata_sha256 == fixture["labels_sha256"]
    assert model.weights_sha256 == fixture["model_sha256"]
    expected = np.asarray(fixture["predictions"])
    actual = model.predict_raw_on_strings(sequences)
    # Loose enough to absorb TensorFlow-vs-PyTorch backend drift, tight
    # enough to catch a wrong checkpoint or a transposed input.
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=4e-5)
    public_pytorch = crested.tl.predict(sequences, model.model, verbose=0)
    np.testing.assert_allclose(actual, public_pytorch, rtol=1e-5, atol=1e-6)
    x = one_hot([sequences[0]], model.sequence_length).requires_grad_(True)
    score = model.inference_on_tensor(x)
    score.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
