"""Integration test: Enformer K562-DNase vs ChromBPNet-K562 correlation.

Scores 100 real genomic sequences (196,608 bp each) with both models and
asserts on Pearson and Spearman correlations.

Both models see the same genomic center region:
  Enformer    receives the full 196,608 bp sequence; K562-DNase tracks,
              bins 436-459 (center 3,000 bp window).
  ChromBPNet  receives the center 2,114 bp crop of each 196,608 bp sequence.

Score caches are shared with the other correlation tests:
  integration_tests/cache/start_seq_enformer/enformer_scores.csv
  integration_tests/cache/start_seq_enformer/chrombpnet_k562_scores.csv

Scatter plot artifact:
  integration_tests/plots/enformer_chrombpnet_k562_scatter.png

To run:
    pytest -s integration_tests/enformer_chrombpnet_k562_correlation_test.py
"""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from tqdm import tqdm

from integration_tests.data_loaders import EnformerStartSequences
from nucleobench.models.chrombpnet.model_def import ChromBPNetOracle
from nucleobench.models.grelu.enformer.model_def import Enformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENFORMER_SEQ_LEN = 196_608
ENFORMER_BIN_SIZE = 128
ENFORMER_CONTEXT_PAD = (ENFORMER_SEQ_LEN - 896 * ENFORMER_BIN_SIZE) // 2  # 40_960
BPNET_SEQ_LEN = 3_000  # reference window used for bin selection
CROP_START_REF = (ENFORMER_SEQ_LEN - BPNET_SEQ_LEN) // 2  # 96_804
CROP_END_REF = CROP_START_REF + BPNET_SEQ_LEN  # 99_804
BIN_FIRST = (CROP_START_REF - ENFORMER_CONTEXT_PAD) // ENFORMER_BIN_SIZE  # 436
BIN_LAST = math.ceil((CROP_END_REF - ENFORMER_CONTEXT_PAD) / ENFORMER_BIN_SIZE)  # 460
SPATIAL_BINS = list(range(BIN_FIRST, BIN_LAST))  # 24 bins

CHROMBPNET_SEQ_LEN = 2_114
CHROMBPNET_CROP_START = (ENFORMER_SEQ_LEN - CHROMBPNET_SEQ_LEN) // 2  # 97_247
CHROMBPNET_CROP_END = CHROMBPNET_CROP_START + CHROMBPNET_SEQ_LEN  # 99_361

N_SEQUENCES = 100

_CACHE_DIR = Path(__file__).parent / "cache" / "start_seq_enformer"
_ENFORMER_SCORES_CSV = _CACHE_DIR / "enformer_scores.csv"
_CHROMBPNET_SCORES_CSV = _CACHE_DIR / "chrombpnet_k562_scores.csv"
_PLOTS_DIR = Path(__file__).parent / "plots"

# Minimum acceptable correlations (slightly below the golden values).
_MIN_PEARSON_R = 0.80
_MIN_SPEARMAN_RHO = 0.80


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scored_sequences():
    """Score all 100 sequences with Enformer and ChromBPNet-K562.

    Reads from the shared CSV caches if available; otherwise runs whichever
    model(s) are missing and writes the results to the cache files.
    """
    import pandas as pd

    if _ENFORMER_SCORES_CSV.exists() and _CHROMBPNET_SCORES_CSV.exists():
        print(f"\nLoading cached scores from {_CACHE_DIR}")
        enformer_scores = pd.read_csv(_ENFORMER_SCORES_CSV)["enformer_score"].to_numpy()
        chrombpnet_scores = pd.read_csv(_CHROMBPNET_SCORES_CSV)[
            "chrombpnet_score"
        ].to_numpy()
        print(f"  enformer_scores shape:   {enformer_scores.shape}")
        print(f"  chrombpnet_scores shape: {chrombpnet_scores.shape}")
        return {
            "enformer_scores": enformer_scores,
            "chrombpnet_scores": chrombpnet_scores,
        }

    # --- Load sequences (only needed if either cache is missing) ---
    print("\nLoading Enformer start sequences...")
    loader = EnformerStartSequences()
    seqs_df = loader.get_data()
    sequences = seqs_df["sequence"].tolist()
    assert len(sequences) == N_SEQUENCES, (
        f"Expected {N_SEQUENCES}, got {len(sequences)}"
    )
    print(f"  Loaded {len(sequences)} sequences ({len(sequences[0])} bp each).")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Enformer ---
    if _ENFORMER_SCORES_CSV.exists():
        print(f"\nLoading cached Enformer scores from {_ENFORMER_SCORES_CSV}")
        enformer_scores = pd.read_csv(_ENFORMER_SCORES_CSV)["enformer_score"].to_numpy()
    else:
        print(f"\nLoading Enformer (k562_dnase, {len(SPATIAL_BINS)} bins)...")
        enformer = Enformer(
            aggregation_type="k562_dnase",
            spatial_bins_to_aggregate=SPATIAL_BINS,
            run_sanity_checks=False,
        )
        print("  Enformer loaded.")
        print("  Scoring with Enformer...")
        enformer_scores = []
        for seq in tqdm(sequences, desc="Enformer"):
            score = enformer([seq]).item()
            enformer_scores.append(-score)
        enformer_scores = np.array(enformer_scores, dtype=np.float64)
        print(
            f"  Enformer scores: min={enformer_scores.min():.3f}"
            f"  max={enformer_scores.max():.3f}"
        )
        pd.DataFrame({"enformer_score": enformer_scores}).to_csv(
            _ENFORMER_SCORES_CSV, index=False
        )
        print(f"  Enformer scores written to {_ENFORMER_SCORES_CSV}")

    # --- ChromBPNet-K562 ---
    if _CHROMBPNET_SCORES_CSV.exists():
        print(f"\nLoading cached ChromBPNet scores from {_CHROMBPNET_SCORES_CSV}")
        chrombpnet_scores = pd.read_csv(_CHROMBPNET_SCORES_CSV)[
            "chrombpnet_score"
        ].to_numpy()
    else:
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
            chrombpnet_scores.append(-score)
        chrombpnet_scores = np.array(chrombpnet_scores, dtype=np.float64)
        print(
            f"  ChromBPNet scores: min={chrombpnet_scores.min():.3f}"
            f"  max={chrombpnet_scores.max():.3f}"
        )
        pd.DataFrame({"chrombpnet_score": chrombpnet_scores}).to_csv(
            _CHROMBPNET_SCORES_CSV, index=False
        )
        print(f"  ChromBPNet scores written to {_CHROMBPNET_SCORES_CSV}")

    return {"enformer_scores": enformer_scores, "chrombpnet_scores": chrombpnet_scores}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.correlation
def test_n_sequences(scored_sequences):
    assert len(scored_sequences["enformer_scores"]) == N_SEQUENCES
    assert len(scored_sequences["chrombpnet_scores"]) == N_SEQUENCES


@pytest.mark.correlation
def test_pearson_r(scored_sequences):
    x = scored_sequences["enformer_scores"]
    y = scored_sequences["chrombpnet_scores"]
    r, p = stats.pearsonr(x, y)
    print(f"\nPearson r = {r:.6f}  (p = {p:.3e})")
    assert r >= _MIN_PEARSON_R, (
        f"Pearson r {r:.4f} is below the threshold {_MIN_PEARSON_R}"
    )


@pytest.mark.correlation
def test_spearman_rho(scored_sequences):
    x = scored_sequences["enformer_scores"]
    y = scored_sequences["chrombpnet_scores"]
    rho, p = stats.spearmanr(x, y)
    print(f"\nSpearman rho = {rho:.6f}  (p = {p:.3e})")
    assert rho >= _MIN_SPEARMAN_RHO, (
        f"Spearman rho {rho:.4f} is below the threshold {_MIN_SPEARMAN_RHO}"
    )


@pytest.mark.correlation
def test_scatter_plot(scored_sequences):
    """Save a scatter plot of Enformer K562-DNase vs ChromBPNet-K562 scores.

    No assertion — this test always passes and produces a plot artifact.
    """
    import matplotlib.pyplot as plt

    x = scored_sequences["enformer_scores"]
    y = scored_sequences["chrombpnet_scores"]

    r, _ = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, alpha=0.7, edgecolors="none", s=40)
    ax.set_xlabel("Enformer K562-DNase score")
    ax.set_ylabel("ChromBPNet K562-ENCSR483RKN score")
    ax.set_title("Enformer K562-DNase vs ChromBPNet-K562\n(100 genomic sequences)")
    ax.annotate(
        f"Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}",
        xy=(0.05, 0.92),
        xycoords="axes fraction",
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _PLOTS_DIR / "enformer_chrombpnet_k562_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nScatter plot saved to {out_path}")
