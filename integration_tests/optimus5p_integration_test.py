"""
Integration test for Optimus 5-Prime model.

To run:
```zsh
pytest -s integration_tests/optimus5p_integration_test.py
```
"""

import random
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import requests
from scipy.stats import spearmanr
from tqdm import tqdm

from nucleobench.models.rna.optimus5p.model_def import Optimus5P
from nucleobench.models.rna.rinalmo_mrl.model_def import RinalmoMRL

GOLDEN_SET_URL = "https://s3.eu-central-1.amazonaws.com/kipoi-models/predictions/individual/5UtrMPRA/expect.human_utrs.h5"


def download_file(url, filename):
    filename = Path(filename)
    if filename.exists():
        return

    print(f"Downloading {url} to {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    filename.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(filename, "wb") as f,
        tqdm(total=total_size, unit="iB", unit_scale=True) as t,
    ):
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)


def one_hot_to_seq(one_hot):
    # Map: A, C, G, T
    bases = ["A", "C", "G", "T"]
    seq = []
    for row in one_hot:
        idx = np.argmax(row)
        if row[idx] == 0:
            seq.append("N")
        else:
            seq.append(bases[idx])
    return "".join(seq)


@pytest.mark.optimus5p
def test_optimus5p_accuracy_vs_kipoi_golden_set():
    """Verify Optimus 5-Prime against Kipoi golden set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        golden_set_path = Path(tmpdir) / "expect.human_utrs.h5"
        download_file(GOLDEN_SET_URL, golden_set_path)

        # 1. Initialize model
        model = Optimus5P()

        # 2. Load Golden Set
        with h5py.File(golden_set_path, "r") as f:
            inputs = f["inputs"][:]  # (2000, 50, 4)
            expected_preds = f["preds"][:].flatten()  # (2000,)

        # 3. Prepare sequences
        sequences = [one_hot_to_seq(inp) for inp in inputs]

        # 4. Run prediction
        # Since our model_def returns -1 * preds to minimize, we multiply by -1 to get original predictions
        actual_preds = -1 * model(sequences)

        # 5. Assertion
        diffs = np.abs(actual_preds - expected_preds)
        max_diff = np.max(diffs)

        assert max_diff < 1e-4, f"Max difference {max_diff} exceeds tolerance 1e-4"


@pytest.mark.optimus5p
def test_optimus5p_rinalmo_correlation_random_data():
    """Generates random DNA sequences and checks correlation between RiNALMo and Optimus."""
    count = 100
    print(f"\nRunning sanity check on {count} random sequences...")

    # Initialize models
    rinalmo_model = RinalmoMRL()
    optimus_model = Optimus5P()
    print("Models initialized.")

    random_seqs = ["".join(random.choices("ACGT", k=100)) for _ in range(count)]

    # model_def already returns negative predictions (to minimize),
    # so we multiply by -1 to get the positive scores for correlation.
    rinalmo_scores = -1 * rinalmo_model(random_seqs)
    optimus_scores = -1 * optimus_model(random_seqs)

    spearman_corr, _ = spearmanr(rinalmo_scores, optimus_scores)
    rinalmo_std = np.std(rinalmo_scores)
    optimus_std = np.std(optimus_scores)

    print(f"Sanity Check Results ({count} random sequences):")
    print(f"  Spearman Correlation: {spearman_corr:.4f}")
    print(f"  Rinalmo Score StdDev: {rinalmo_std:.4f}")
    print(f"  Optimus Score StdDev: {optimus_std:.4f}")

    # Assert Spearman correlation is at least 0.3
    assert spearman_corr > 0.3, (
        f"Spearman correlation {spearman_corr:.4f} is too low (< 0.3)"
    )


@pytest.mark.optimus5p
def test_optimus5p_sensitivity_to_dna_vs_rna():
    """Checks if the model is sensitive to DNA (T) vs RNA (U) inputs."""
    model = Optimus5P()

    # DNA (with Ts) - 50bp each
    seq_dna_1 = "AAAAACCCCCTTTTTGGGGGAAAAACCCCCTTTTTGGGGGAAAAACCC"
    seq_dna_2 = "GGGGGTTTTTAAAAACCCCCGGGGGTTTTTAAAAACCCCCGGGGGTTT"

    # RNA (with Us)
    seq_rna_1 = seq_dna_1.replace("T", "U")
    seq_rna_2 = seq_dna_2.replace("T", "U")

    score_d1 = -1 * model([seq_dna_1])[0]
    score_d2 = -1 * model([seq_dna_2])[0]
    diff_dna = abs(score_d1 - score_d2)

    score_r1 = -1 * model([seq_rna_1])[0]
    score_r2 = -1 * model([seq_rna_2])[0]
    diff_rna = abs(score_r1 - score_r2)

    assert diff_dna > 0.1, "Model is insensitive to DNA (Ts)"
    assert diff_rna > 0.1, "Model is insensitive to RNA (Us)"
    assert abs(score_d1 - score_r1) < 1e-5, (
        "DNA and RNA inputs should yield identical predictions"
    )


@pytest.mark.optimus5p
def test_optimus5p_dynamic_range():
    """Checks if the model has a reasonable dynamic range on synthetic controls."""
    model = Optimus5P()

    # GOOD: All As (High efficiency, no secondary structure, no ATGs)
    seq_good = "A" * 50

    # BAD: ATG repeats (Strong repression)
    seq_bad = "ATG" * 16 + "AT"

    score_good = -1 * model([seq_good])[0]
    score_bad = -1 * model([seq_bad])[0]
    diff = abs(score_good - score_bad)

    assert diff > 1.0, f"Model lacks dynamic range (diff={diff:.4f} <= 1.0)"


if __name__ == "__main__":
    pytest.main([__file__])
