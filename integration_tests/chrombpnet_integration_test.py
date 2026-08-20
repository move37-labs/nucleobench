"""Integration tests for ChromBPNetOracle.

Requires network on first run (downloads Adrenal_c0.gz from Zenodo, caches
fold-0 chrombpnet_nobias.h5). Subsequent runs use ~/.cache/nucleobench/chrombpnet/.

To run:
    pytest -s -m chrombpnet integration_tests/chrombpnet_integration_test.py
"""

import pytest

from nucleobench.models.chrombpnet import constants as cb_constants
from nucleobench.models.chrombpnet.model_def import ChromBPNetOracle

_L = cb_constants.SEQ_LEN
POLYA = "A" * _L
GC_RICH = ("GC" * (_L // 2 + 1))[:_L]


@pytest.fixture(scope="module")
def oracle():
    return ChromBPNetOracle(cell_type="Adrenal_c0")


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
def test_raises_on_string_not_list(oracle):
    with pytest.raises(ValueError, match="list of strings"):
        oracle(POLYA)
