"""Integration test: Enformer (K562-DNase) vs BPNet-ATAC correlation.

Scores 100 real genomic sequences (196,608 bp each) with both models and
asserts Pearson r >= 0.73 and Spearman rho >= 0.87.

Expected golden values (from original scoring run):
  Pearson  r   = 0.7312583504871208  (p = 5.68e-18)
  Spearman rho = 0.8753555355535554  (p = 1.07e-32)

Score caches (written on first run, reused on subsequent runs):
  integration_tests/cache/start_seq_enformer/enformer_scores.csv
  integration_tests/cache/start_seq_enformer/bpnet_atac_scores.csv

Scatter plot artifact:
  integration_tests/plots/enformer_bpnet_atac_scatter.png

To run:
    pytest -s -m enformer_bpnet_atac \\
        integration_tests/enformer_bpnet_atac_correlation_test.py
"""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from tqdm import tqdm

from integration_tests.data_loaders import EnformerStartSequences
from nucleobench.models.bpnet.model_def import BPNet
from nucleobench.models.grelu.enformer import constants as enf_constants
from nucleobench.models.grelu.enformer.model_def import Enformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENFORMER_SEQ_LEN = 196_608
BPNET_SEQ_LEN = 3_000
ENFORMER_BIN_SIZE = 128
ENFORMER_CONTEXT_PAD = (ENFORMER_SEQ_LEN - 896 * ENFORMER_BIN_SIZE) // 2  # 40_960
CROP_START = (ENFORMER_SEQ_LEN - BPNET_SEQ_LEN) // 2  # 96_804
CROP_END = CROP_START + BPNET_SEQ_LEN  # 99_804
BIN_FIRST = (CROP_START - ENFORMER_CONTEXT_PAD) // ENFORMER_BIN_SIZE  # 436
BIN_LAST = math.ceil((CROP_END - ENFORMER_CONTEXT_PAD) / ENFORMER_BIN_SIZE)  # 460
SPATIAL_BINS = list(range(BIN_FIRST, BIN_LAST))  # 24 bins

N_SEQUENCES = 100

_CACHE_DIR = Path(__file__).parent / "cache" / "start_seq_enformer"
_ENFORMER_SCORES_CSV = _CACHE_DIR / "enformer_scores.csv"
_BPNET_SCORES_CSV = _CACHE_DIR / "bpnet_atac_scores.csv"
_PLOTS_DIR = Path(__file__).parent / "plots"

# Minimum acceptable correlations (slightly below the golden values).
_MIN_PEARSON_R = 0.73
_MIN_SPEARMAN_RHO = 0.87


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scored_sequences():
    """Score all 100 sequences with Enformer and BPNet-ATAC.

    Each model's scores are cached independently. If a model's CSV already
    exists it is loaded directly; only missing models are run.
    """
    import pandas as pd

    if _ENFORMER_SCORES_CSV.exists() and _BPNET_SCORES_CSV.exists():
        print(f"\nLoading cached scores from {_CACHE_DIR}")
        enformer_scores = pd.read_csv(_ENFORMER_SCORES_CSV)["enformer_score"].to_numpy()
        bpnet_scores = pd.read_csv(_BPNET_SCORES_CSV)["bpnet_atac_score"].to_numpy()
        print(f"  enformer_scores shape: {enformer_scores.shape}")
        print(f"  bpnet_scores shape:    {bpnet_scores.shape}")
        return {"enformer_scores": enformer_scores, "bpnet_scores": bpnet_scores}

    # --- Load sequences (only needed if either cache is missing) ---
    print("\nLoading Enformer start sequences...")
    loader = EnformerStartSequences()
    seqs_df = loader.get_data()
    sequences = seqs_df["sequence"].tolist()
    assert len(sequences) == N_SEQUENCES, f"Expected {N_SEQUENCES}, got {len(sequences)}"
    print(f"  Loaded {len(sequences)} sequences ({len(sequences[0])} bp each).")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Enformer ---
    if _ENFORMER_SCORES_CSV.exists():
        print(f"\nLoading cached Enformer scores from {_ENFORMER_SCORES_CSV}")
        enformer_scores = pd.read_csv(_ENFORMER_SCORES_CSV)["enformer_score"].to_numpy()
    else:
        track_idxs = enf_constants.k562_dnase_track_indices()
        print(
            f"\nLoading Enformer (k562_dnase, {len(track_idxs)} tracks,"
            f" {len(SPATIAL_BINS)} bins)..."
        )
        enformer = Enformer(
            aggregation_type="k562_dnase",
            track_indices=track_idxs,
            spatial_bins_to_aggregate=SPATIAL_BINS,
            run_sanity_checks=False,
        )
        print("  Enformer loaded.")
        print("  Scoring with Enformer...")
        enformer_scores = []
        for seq in tqdm(sequences, desc="Enformer"):
            score = enformer([seq]).item()
            # Enformer wrapper negates (minimization); flip back to raw signal.
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

    # --- BPNet-ATAC ---
    if _BPNET_SCORES_CSV.exists():
        print(f"\nLoading cached BPNet-ATAC scores from {_BPNET_SCORES_CSV}")
        bpnet_scores = pd.read_csv(_BPNET_SCORES_CSV)["bpnet_atac_score"].to_numpy()
    else:
        print("\nLoading BPNet-ATAC...")
        bpnet = BPNet(protein="ATAC")
        print("  BPNet-ATAC loaded.")
        print("  Scoring with BPNet-ATAC...")
        bpnet_scores = []
        for seq in tqdm(sequences, desc="BPNet-ATAC"):
            crop = seq[CROP_START:CROP_END]
            assert len(crop) == BPNET_SEQ_LEN, f"Crop length {len(crop)} != {BPNET_SEQ_LEN}"
            score = bpnet([crop]).item()
            # BPNet wrapper negates (minimization); flip back to raw signal.
            bpnet_scores.append(-score)
        bpnet_scores = np.array(bpnet_scores, dtype=np.float64)
        print(f"  BPNet scores: min={bpnet_scores.min():.3f}  max={bpnet_scores.max():.3f}")
        pd.DataFrame({"bpnet_atac_score": bpnet_scores}).to_csv(
            _BPNET_SCORES_CSV, index=False
        )
        print(f"  BPNet-ATAC scores written to {_BPNET_SCORES_CSV}")

    return {"enformer_scores": enformer_scores, "bpnet_scores": bpnet_scores}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.enformer_bpnet_atac
def test_n_sequences(scored_sequences):
    assert len(scored_sequences["enformer_scores"]) == N_SEQUENCES
    assert len(scored_sequences["bpnet_scores"]) == N_SEQUENCES


@pytest.mark.enformer_bpnet_atac
def test_pearson_r(scored_sequences):
    x = scored_sequences["bpnet_scores"]
    y = scored_sequences["enformer_scores"]
    r, p = stats.pearsonr(x, y)
    print(f"\nPearson r = {r:.6f}  (p = {p:.3e})")
    assert r >= _MIN_PEARSON_R, (
        f"Pearson r {r:.4f} is below the threshold {_MIN_PEARSON_R}"
    )


@pytest.mark.enformer_bpnet_atac
def test_spearman_rho(scored_sequences):
    x = scored_sequences["bpnet_scores"]
    y = scored_sequences["enformer_scores"]
    rho, p = stats.spearmanr(x, y)
    print(f"\nSpearman rho = {rho:.6f}  (p = {p:.3e})")
    assert rho >= _MIN_SPEARMAN_RHO, (
        f"Spearman rho {rho:.4f} is below the threshold {_MIN_SPEARMAN_RHO}"
    )


@pytest.mark.enformer_bpnet_atac
def test_scatter_plot(scored_sequences):
    """Save a scatter plot of BPNet-ATAC vs Enformer scores.

    No assertion — this test always passes and produces a plot artifact.
    """
    import matplotlib.pyplot as plt

    x = scored_sequences["bpnet_scores"]
    y = scored_sequences["enformer_scores"]

    r, _ = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, alpha=0.7, edgecolors="none", s=40)
    ax.set_xlabel("BPNet-ATAC score")
    ax.set_ylabel("Enformer K562-DNase score")
    ax.set_title("BPNet-ATAC vs Enformer K562-DNase\n(100 genomic sequences)")
    ax.annotate(
        f"Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}",
        xy=(0.05, 0.92),
        xycoords="axes fraction",
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _PLOTS_DIR / "enformer_bpnet_atac_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nScatter plot saved to {out_path}")
