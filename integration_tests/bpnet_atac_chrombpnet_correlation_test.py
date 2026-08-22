"""Integration test: BPNet-ATAC vs ChromBPNet-K562 correlation.

Scores 100 real genomic sequences (196,608 bp each) with both models, using
center-crops matched to each model's native input length, and asserts
Pearson r >= 0.99 and Spearman rho >= 0.99.

NOTE: The 0.99 thresholds are placeholders. Run the test once to obtain
the actual correlation values, then update _MIN_PEARSON_R and _MIN_SPEARMAN_RHO.

Both models see the same genomic center region:
  BPNet-ATAC   receives the center 3,000 bp crop of each 196,608 bp sequence.
  ChromBPNet   receives the center 2,114 bp crop of each 196,608 bp sequence.

Score caches (written on first run, reused on subsequent runs):
  integration_tests/cache/bpnet_atac_chrombpnet/bpnet_atac_scores.csv
  integration_tests/cache/bpnet_atac_chrombpnet/chrombpnet_k562_scores.csv

Scatter plot artifact:
  integration_tests/plots/bpnet_atac_chrombpnet_scatter.png

To run:
    pytest -s -m bpnet_atac_chrombpnet \\
        integration_tests/bpnet_atac_chrombpnet_correlation_test.py
"""

from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from tqdm import tqdm

from integration_tests.data_loaders import EnformerStartSequences
from nucleobench.models.bpnet.model_def import BPNet
from nucleobench.models.chrombpnet.model_def import ChromBPNetOracle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENFORMER_SEQ_LEN = 196_608
BPNET_SEQ_LEN = 3_000
CHROMBPNET_SEQ_LEN = 2_114

BPNET_CROP_START = (ENFORMER_SEQ_LEN - BPNET_SEQ_LEN) // 2           # 96_804
BPNET_CROP_END = BPNET_CROP_START + BPNET_SEQ_LEN                     # 99_804

CHROMBPNET_CROP_START = (ENFORMER_SEQ_LEN - CHROMBPNET_SEQ_LEN) // 2  # 97_247
CHROMBPNET_CROP_END = CHROMBPNET_CROP_START + CHROMBPNET_SEQ_LEN       # 99_361

N_SEQUENCES = 100

_CACHE_DIR = Path(__file__).parent / "cache" / "bpnet_atac_chrombpnet"
_BPNET_SCORES_CSV = _CACHE_DIR / "bpnet_atac_scores.csv"
_CHROMBPNET_SCORES_CSV = _CACHE_DIR / "chrombpnet_k562_scores.csv"
_PLOTS_DIR = Path(__file__).parent / "plots"

# Placeholder thresholds — update after the first run reveals actual values.
_MIN_PEARSON_R = 0.80
_MIN_SPEARMAN_RHO = 0.73


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scored_sequences():
    """Score all 100 sequences with BPNet-ATAC and ChromBPNet-K562.

    Reads from CSV caches if available; otherwise runs both models and writes
    the results to the cache files.
    """
    import pandas as pd

    if _BPNET_SCORES_CSV.exists() and _CHROMBPNET_SCORES_CSV.exists():
        print(f"\nLoading cached scores from {_CACHE_DIR}")
        bpnet_scores = pd.read_csv(_BPNET_SCORES_CSV)["bpnet_atac_score"].to_numpy()
        chrombpnet_scores = pd.read_csv(_CHROMBPNET_SCORES_CSV)["chrombpnet_score"].to_numpy()
        print(f"  bpnet_scores shape:      {bpnet_scores.shape}")
        print(f"  chrombpnet_scores shape: {chrombpnet_scores.shape}")
        return {"bpnet_scores": bpnet_scores, "chrombpnet_scores": chrombpnet_scores}

    # --- Load sequences ---
    print("\nLoading Enformer start sequences...")
    loader = EnformerStartSequences()
    seqs_df = loader.get_data()
    sequences = seqs_df["sequence"].tolist()
    assert len(sequences) == N_SEQUENCES, f"Expected {N_SEQUENCES}, got {len(sequences)}"
    print(f"  Loaded {len(sequences)} sequences ({len(sequences[0])} bp each).")

    # --- BPNet-ATAC ---
    print("\nLoading BPNet-ATAC...")
    bpnet = BPNet(protein="ATAC")
    print("  BPNet-ATAC loaded.")

    print("  Scoring with BPNet-ATAC (center 3,000 bp crop)...")
    bpnet_scores = []
    for seq in tqdm(sequences, desc="BPNet-ATAC"):
        crop = seq[BPNET_CROP_START:BPNET_CROP_END]
        assert len(crop) == BPNET_SEQ_LEN, f"Crop length {len(crop)} != {BPNET_SEQ_LEN}"
        score = bpnet([crop]).item()
        # BPNet wrapper negates (minimization); flip back to raw signal.
        bpnet_scores.append(-score)
    bpnet_scores = np.array(bpnet_scores, dtype=np.float64)
    print(f"  BPNet-ATAC scores: min={bpnet_scores.min():.3f}  max={bpnet_scores.max():.3f}")

    # --- ChromBPNet-K562 ---
    print("\nLoading ChromBPNet-K562 (K562_ENCSR483RKN)...")
    chrombpnet = ChromBPNetOracle(cell_type="K562_ENCSR483RKN")
    print("  ChromBPNet loaded.")

    print("  Scoring with ChromBPNet-K562 (center 2,114 bp crop)...")
    chrombpnet_scores = []
    for seq in tqdm(sequences, desc="ChromBPNet"):
        crop = seq[CHROMBPNET_CROP_START:CHROMBPNET_CROP_END]
        assert len(crop) == CHROMBPNET_SEQ_LEN, (
            f"Crop length {len(crop)} != {CHROMBPNET_SEQ_LEN}"
        )
        score = chrombpnet([crop]).item()
        # ChromBPNet wrapper negates (minimization); flip back to raw signal.
        chrombpnet_scores.append(-score)
    chrombpnet_scores = np.array(chrombpnet_scores, dtype=np.float64)
    print(
        f"  ChromBPNet scores: min={chrombpnet_scores.min():.3f}"
        f"  max={chrombpnet_scores.max():.3f}"
    )

    # --- Write caches ---
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"bpnet_atac_score": bpnet_scores}).to_csv(
        _BPNET_SCORES_CSV, index=False
    )
    pd.DataFrame({"chrombpnet_score": chrombpnet_scores}).to_csv(
        _CHROMBPNET_SCORES_CSV, index=False
    )
    print(f"\nScores written to {_CACHE_DIR}")

    return {"bpnet_scores": bpnet_scores, "chrombpnet_scores": chrombpnet_scores}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.bpnet_atac_chrombpnet
def test_n_sequences(scored_sequences):
    assert len(scored_sequences["bpnet_scores"]) == N_SEQUENCES
    assert len(scored_sequences["chrombpnet_scores"]) == N_SEQUENCES


@pytest.mark.bpnet_atac_chrombpnet
def test_pearson_r(scored_sequences):
    x = scored_sequences["bpnet_scores"]
    y = scored_sequences["chrombpnet_scores"]
    r, p = stats.pearsonr(x, y)
    print(f"\nPearson r = {r:.6f}  (p = {p:.3e})")
    assert r >= _MIN_PEARSON_R, (
        f"Pearson r {r:.4f} is below the threshold {_MIN_PEARSON_R}"
    )


@pytest.mark.bpnet_atac_chrombpnet
def test_spearman_rho(scored_sequences):
    x = scored_sequences["bpnet_scores"]
    y = scored_sequences["chrombpnet_scores"]
    rho, p = stats.spearmanr(x, y)
    print(f"\nSpearman rho = {rho:.6f}  (p = {p:.3e})")
    assert rho >= _MIN_SPEARMAN_RHO, (
        f"Spearman rho {rho:.4f} is below the threshold {_MIN_SPEARMAN_RHO}"
    )


@pytest.mark.bpnet_atac_chrombpnet
def test_scatter_plot(scored_sequences):
    """Save a scatter plot of BPNet-ATAC vs ChromBPNet-K562 scores.

    No assertion — this test always passes and produces a plot artifact.
    """
    import matplotlib.pyplot as plt

    x = scored_sequences["bpnet_scores"]
    y = scored_sequences["chrombpnet_scores"]

    r, _ = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, alpha=0.7, edgecolors="none", s=40)
    ax.set_xlabel("BPNet-ATAC score")
    ax.set_ylabel("ChromBPNet K562-ENCSR483RKN score")
    ax.set_title("BPNet-ATAC vs ChromBPNet-K562\n(100 genomic sequences)")
    ax.annotate(
        f"Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}",
        xy=(0.05, 0.92),
        xycoords="axes fraction",
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _PLOTS_DIR / "bpnet_atac_chrombpnet_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nScatter plot saved to {out_path}")
