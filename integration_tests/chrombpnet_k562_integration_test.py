"""Integration tests for ChromBPNetOracle with K562_ENCSR483RKN weights.

Downloads fold-0 chrombpnet_nobias model from HuggingFace on first run
(kundajelab/encode-chrombpnet-ATAC-K562-ENCSR483RKN-ENCSR780QKO); subsequent
runs use HuggingFace's own cache (~/.cache/huggingface/hub/).

Golden correctness test requires testdata/k562_encsr483rkn_goldens.json.
Generate it once with TF/Keras via:
    python nucleobench/models/chrombpnet/generate_k562_goldens.py \\
        --out nucleobench/models/chrombpnet/testdata/k562_encsr483rkn_goldens.json
(Docker recommended — see that script's docstring for a one-liner.)

To run:
    pytest -s -m chrombpnet_k562 integration_tests/chrombpnet_k562_integration_test.py
"""

import json
from pathlib import Path

import pytest

from nucleobench.models.chrombpnet import constants as cb_constants
from nucleobench.models.chrombpnet.model_def import ChromBPNetOracle

_L = cb_constants.SEQ_LEN
POLYA = "A" * _L
GC_RICH = ("GC" * (_L // 2 + 1))[:_L]

_TESTDATA_DIR = (
    Path(__file__).parent.parent / "nucleobench" / "models" / "chrombpnet" / "testdata"
)
_GOLDEN_PATH = _TESTDATA_DIR / "k562_encsr483rkn_goldens.json"


@pytest.fixture(scope="module")
def oracle():
    return ChromBPNetOracle(cell_type="K562_ENCSR483RKN")


@pytest.mark.chrombpnet_k562
def test_output_shape(oracle):
    result = oracle([POLYA, GC_RICH])
    assert result.shape == (2,)


@pytest.mark.chrombpnet_k562
def test_scores_are_negative(oracle):
    scores = oracle([POLYA, GC_RICH])
    assert (scores < 0).all(), f"Expected all negative, got: {scores}"


@pytest.mark.chrombpnet_k562
def test_dynamic_range(oracle):
    scores = oracle([POLYA, GC_RICH])
    assert scores[0] != scores[1], f"Expected distinct scores, got {scores}"


@pytest.mark.chrombpnet_k562
def test_raises_on_short_sequence(oracle):
    with pytest.raises(ValueError, match="SEQ_LEN"):
        oracle(["ACGT"])


@pytest.mark.chrombpnet_k562
def test_raises_on_string_not_list(oracle):
    with pytest.raises(ValueError, match="list of strings"):
        oracle(POLYA)


@pytest.mark.chrombpnet_k562
def test_pytorch_matches_keras_goldens(oracle):
    """Assert bpnet-lite PyTorch conversion matches official TF/Keras predictions.

    Goldens are generated once via
    nucleobench/models/chrombpnet/generate_k562_goldens.py and checked in.
    atol=1e-5 matches the tolerance used for Adrenal_c0 (max observed diff
    is ~4.77e-07 for that model; calibrate with --also-pytorch if needed).
    """
    if not _GOLDEN_PATH.exists():
        pytest.fail(
            f"Golden file not found: {_GOLDEN_PATH}. "
            "Generate it with nucleobench/models/chrombpnet/generate_k562_goldens.py"
        )

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
