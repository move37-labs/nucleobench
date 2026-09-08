"""Integration tests for ChromBPNetOracle.

Requires network on first run (downloads Adrenal_c0.gz from Zenodo, caches
fold-0 chrombpnet_nobias.h5). Subsequent runs use ~/.cache/nucleobench/chrombpnet/.
Biological ranking also fetches 2114-bp hg38 windows from the UCSC REST API.

To run:
    pytest -s -m chrombpnet integration_tests/chrombpnet_integration_test.py
"""

import json
from pathlib import Path

import pytest
import requests

from nucleobench.models.chrombpnet import constants as cb_constants
from nucleobench.models.chrombpnet.model_def import ChromBPNetOracle

_L = cb_constants.SEQ_LEN
POLYA = "A" * _L
GC_RICH = ("GC" * (_L // 2 + 1))[:_L]

_TESTDATA_DIR = (
    Path(__file__).parent.parent / "nucleobench" / "models" / "chrombpnet" / "testdata"
)
_GOLDEN_PATH = _TESTDATA_DIR / "adrenal_c0_keras_goldens.json"

# hg38 1-based centers; windows use the same centering as Enformer
# (MuscleGeneExpressionByBucket._extract_sequence, style="enformer").
# CYP11A1: adrenal steroidogenesis gene, Ensembl ENSG00000140459, minus
# strand, TSS = gene end. Do not use HDMA Brain_c0 tutorial loci here.
_CYP11A1_CHROM = "chr15"
_CYP11A1_TSS = 74_367_884
# Same gene-desert locus as the Enformer / Borzoi integration tests.
_GENE_DESERT_CHROM = "chr8"
_GENE_DESERT_CENTER = 127_150_000

_UCSC_SEQUENCE_URL = (
    "https://api.genome.ucsc.edu/getData/sequence"
    "?genome=hg38;chrom={chrom};start={start};end={end}"
)


def _enformer_style_window(center: int, length: int) -> tuple[int, int]:
    """Return 1-based inclusive start, end for a length-bp window at center."""
    start = center - (length // 2)
    end = center + ((length - 1) // 2)
    return start, end


def _fetch_hg38(chrom: str, center: int, length: int = _L) -> str:
    """Fetch a centered hg38 window from the UCSC REST API."""
    start, end = _enformer_style_window(center, length)
    url = _UCSC_SEQUENCE_URL.format(chrom=chrom, start=start - 1, end=end)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    seq = response.json()["dna"].upper()
    if len(seq) != length:
        raise ValueError(
            f"Expected {length} bp from {chrom}:{start}-{end}, got {len(seq)}"
        )
    if any(nt not in "ACGT" for nt in seq):
        raise ValueError(f"Non-ACGT bases in {chrom}:{start}-{end}")
    return seq


@pytest.fixture(scope="module")
def oracle():
    return ChromBPNetOracle(cell_type="Adrenal_c0")


@pytest.fixture(scope="module")
def ranking_preds(oracle):
    """Accessibility scores (negated oracle output) for ranking checks.

    The oracle minimizes predicted counts, so raw scores are negative.
    Flip the sign so higher means more predicted accessibility, matching
    the Enformer integration test.
    """
    cyp11a1 = _fetch_hg38(_CYP11A1_CHROM, _CYP11A1_TSS)
    gene_desert = _fetch_hg38(_GENE_DESERT_CHROM, _GENE_DESERT_CENTER)
    cyp_acc, desert_acc, polya_acc = -oracle([cyp11a1, gene_desert, POLYA])
    print(
        "\nAdrenal_c0 accessibility (higher = more predicted counts):\n"
        f"  CYP11A1 TSS: {cyp_acc:.4f}\n"
        f"  gene desert: {desert_acc:.4f}\n"
        f"  poly-A:      {polya_acc:.4f}"
    )
    return {
        "cyp11a1": float(cyp_acc),
        "gene_desert": float(desert_acc),
        "polya": float(polya_acc),
    }


@pytest.mark.chrombpnet
def test_output_shape(oracle):
    result = oracle([POLYA, GC_RICH])
    assert result.shape == (2,)


@pytest.mark.chrombpnet
def test_scores_are_negative(oracle):
    scores = oracle([POLYA, GC_RICH])
    assert (scores < 0).all(), f"Expected all negative, got: {scores}"


@pytest.mark.chrombpnet
def test_dynamic_range(oracle):
    scores = oracle([POLYA, GC_RICH])
    assert scores[0] != scores[1], f"Expected distinct scores, got {scores}"


@pytest.mark.chrombpnet
def test_raises_on_short_sequence(oracle):
    with pytest.raises(ValueError, match="SEQ_LEN"):
        oracle(["ACGT"])


@pytest.mark.chrombpnet
def test_3k_input_is_valid(oracle):
    result = oracle(["A" * 3000])
    assert result.shape == (1,)


@pytest.mark.chrombpnet
def test_raises_on_string_not_list(oracle):
    with pytest.raises(ValueError, match="list of strings"):
        oracle(POLYA)


@pytest.mark.chrombpnet
def test_cyp11a1_more_accessible_than_polya(ranking_preds):
    assert ranking_preds["cyp11a1"] > ranking_preds["polya"], (
        "CYP11A1 promoter should be more accessible than poly-A on Adrenal_c0"
    )


@pytest.mark.chrombpnet
def test_cyp11a1_more_accessible_than_gene_desert(ranking_preds):
    assert ranking_preds["cyp11a1"] > ranking_preds["gene_desert"], (
        "CYP11A1 promoter should be more accessible than a gene desert on Adrenal_c0"
    )


@pytest.mark.chrombpnet
def test_pytorch_matches_keras_goldens(oracle):
    """Assert bpnet-lite PyTorch conversion matches official TF/Keras predictions.

    Goldens were generated once with tensorflow:2.11.0 via
    nucleobench/models/chrombpnet/generate_keras_goldens.py and checked in.
    Max |keras - pytorch| measured at 4.77e-07 (float32 accumulation only).
    atol=1e-5 gives ~20x margin.
    """
    with open(_GOLDEN_PATH) as f:
        goldens = json.load(f)

    seqs = [ex["sequence"] for ex in goldens["examples"]]
    keras_counts = [ex["keras_counts"] for ex in goldens["examples"]]
    names = [ex["name"] for ex in goldens["examples"]]

    # oracle negates counts; flip back to raw counts for comparison.
    pytorch_counts = list(-oracle(seqs))

    for name, keras, pytorch in zip(names, keras_counts, pytorch_counts):
        assert pytorch == pytest.approx(keras, abs=1e-5), (
            f"{name}: keras_counts={keras:.6f}  pytorch_counts={pytorch:.6f}  "
            f"|diff|={abs(keras - pytorch):.2e} (atol=1e-5)"
        )
