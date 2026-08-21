"""Generate official TF/Keras ChromBPNet golden I/O pairs for Adrenal_c0.

Run once in a Linux TF environment (not the NucleoBench env, which is PyTorch-only).
The output JSON is checked in and consumed by the integration test.

Usage
-----
# Minimal: uses the already-cached NucleoBench h5
python generate_keras_goldens.py \
    --weights ~/.cache/nucleobench/chrombpnet/Adrenal_c0.h5 \
    --out nucleobench/models/chrombpnet/testdata/adrenal_c0_keras_goldens.json

# Optional: also score via bpnet-lite and report max |keras - pytorch| to set atol
python generate_keras_goldens.py \
    --weights ~/.cache/nucleobench/chrombpnet/Adrenal_c0.h5 \
    --out nucleobench/models/chrombpnet/testdata/adrenal_c0_keras_goldens.json \
    --also-pytorch

Docker one-liner (no macOS arm64 TF wheel):
    docker run --rm \
        -v ~/.cache/nucleobench/chrombpnet:/weights:ro \
        -v $(pwd):/repo \
        tensorflow/tensorflow:2.11.0 \
        python /repo/nucleobench/models/chrombpnet/generate_keras_goldens.py \
            --weights /weights/Adrenal_c0.h5 \
            --out /repo/nucleobench/models/chrombpnet/testdata/adrenal_c0_keras_goldens.json

Environment
-----------
Requires TensorFlow 2.x (tested with tensorflow:2.11.0 Docker image).
Does NOT require the `chrombpnet` pip package; chrombpnet_nobias.h5 uses
only standard Keras layers and loads cleanly without custom objects.
"""

import argparse
import datetime
import json
import os

import numpy as np

# Sequences to score.  The first two are synthetic controls matching the
# integration test smoke checks; the last two are real hg38 loci fetched once
# below and embedded here so the generator does NOT require network access.
SEQ_LEN = 2114

_POLYA = "A" * SEQ_LEN
_GC_RICH = ("GC" * (SEQ_LEN // 2 + 1))[:SEQ_LEN]

# hg38 windows fetched 2026-08-21 from UCSC REST API using the same centering
# formula as _fetch_hg38 in the integration test:
#   start_1based = center - 1057; ucsc_0based_start = start_1based - 1
# CYP11A1 TSS: chr15:74367884 (Ensembl ENSG00000140459, minus strand).
#   UCSC query: chrom=chr15, start=74366826, end=74368940 → 2114 bp.
# Gene desert: chr8:127150000 (same locus as Enformer/Borzoi integration tests).
#   UCSC query: chrom=chr8, start=127148942, end=127151056 → 2114 bp.
_CYP11A1 = ("TGCTTCAGCCTCCCGAGTAGCTGGGATTACAGGCGCCTGCCACCACACCCAGCTAATTTTTGTATTTTTAGTAGAGACGGGGTTTCATCATGTTGGCCAGGATGGTCTCGAACTCTTGATCTCAAGTGATCCGCCCACCTCGGCCTCCCAAAGTGCTGGGATTATGGGTGTGAGCCACCATGCCTGGCCAGACCTGGGGTTTGGACCCATTAAAGAGCTTCCTTACAGATGGCTGGCGGTCTTGCCACCAATTGTTTCAAAAGTAGATGTCTGGTTTAGAGTACTTAAGACATTCACAGATTCAAATGTTGAATTTTGAAATATCCCTGATATATTTCTGTATTGTATTACCAAAAAAAAAAAAAAAAAAGAAGTTAGACAGGAGTTTGGAACCAGAGAACAGCCTGTTGGGGGAGTGGGGACTACAGCAGGGCTACCCAGGCCCCTCCTCCTCCCTGTCCCTTCGGCTCCCACCCTCTGCCAGGCTTACCTGTAAATCGGGCCATACTTCTGGAAATTCTGGACATGGTGAAGGTGGACTTTGTGTGTGCCCGTCTCCCTCCAGAAATGGTACAGGTTTAGCCAGCCATTGTCACCAGGAGAGGGGATCTCATTGAAGGGGCGAGGACTGCGGGTGGAGATGCCAGCTCCCTCGCCAGTGGGCACCCTGAGACGCCCCAGCCCCTCCCTGGGGGCACTCAGAAAGGTCTGGCAGCCTTTGACCAGGACTGAGCGTGGGGGAAGACCCTTGGCCAGCATGCTGTCCCCACAGCTGTGACTGTACCTGCTCCACTTCAGCGGGGACTGCTAGGATGACTGTAGCCCTGGGCCACCAGGGCCAAGATTATAACTACCAGCTCAAGGCCATACCAGAAGCTGATAAAATGTTCACGTCCTTCCTCCTGCTGCAGGCTCAAACCCGCAGACAGCTCCTCCCCTACTCGGTATGAAGGTTCAGTCTGGAATTTCTGCAGCGTGAGGTTCTCCAAAGGACAGGGAGACCCCTCTGTCTCTGCCCTCAGTGCCAGCCCTAGCAGACTCCTGAAATATCTGCCTGTGCTTTTGCCTGAGCTGGGGCCTCCCACTGGCCCCTGCAGCGGAGTCCATGCCCTGGCAGCCATTGGTCAAGACTGCAGACAACCTGTGTTTGGGGGAAATGAGGGGCCCAAGAGAAGGTCAGAGCTATCTTGCCAGCTTGGGCAACATACGTCTTAGTTATGGCCCCACTGTATAGACACACATCATTGAACAACAGGGATAAGTTCTGAGAAATGCATCCTTAGGCTATTTTATTTTATTTTATTTTATTTTATTTTATTTTATTTTAAGAGAGTTTCATTCTTGTTGCCCAGGCTGGAGCGCAATGGCACAGTTTCACCTTGGAACCCTAATCCTCGTCGTCTCTCCCCAGTTGACCCAGATTTGGGGCTGTAATTCAGTACAGCTGACTCAGAGGGGCTCTGGGTCCCTTGAGCTGAGGTCTGTGGTCAGGCTCCACTCTGGACACACTCTAAGACCTTGGCAAATCTGGTTGCCATGGAACTCAGGATGTCAGCTCTTCCCTGTCCACACACTTGATCTGTGCTGCTTTCCCCTCAGAGCTGCTTCCATACCCCCTCCCATCATATCATATCCCCAGACCCATCCTTCCAGCCCAGGTCAGCCCCCACACCCACCCCCCATCACCCTGCACCTACCACTTTCCCAAACAACAAGGGTCAGTCACCTGTAGAGTTTTCAAGTCATTTCAACAAAATTACAGCCCACTTGGTAAAAATGTGTAGGTCAGAAAACATACATGTGCCATGTAGAGCATTTCATGACAGCAATCTACACTATACCTTATACTTATCATTGGAAGGTACAACTTCTTGCTTTTTTTCCCCAAAATCTTATTTTCTGGCAAATAAGGCATTTAAAACACACATCTAAGAGACAGAAAATAGACTGGTGGTTGCTTAGAGCTAGGGGGATAGAGGAATTAGAGGATGGGACATAAAGTGTGTAGGGATTCTTAGGATGATGAAAATGAAAATATTCTAAAGTATATTGTGGTAATTATTGAACAACTCTCTGATACACTAAAAGCCATTGAATTGCATACTTTAAATGGG")

_GENE_DESERT = ("AGTCAAACAGCATTACAGAAGAATATCAGTTATATCAGTTTATCAAACTCTAATTTCCCATGACTATATCAACACACACAACCACAAAAATATAAACCAACTGCTGTAACAACAAGCTCTAAGAGTATCCAAACTGAGGCAGTCAGGGTGCTTCCCTCTCTCAGTTGGGCTTGTTCAACCTATAAATGGAAATTCTTTAAAAAATTTCCCAGCCAGGTGTGGTAGCTCATGCCTGTAATGCCAGCACTTTGGGAGGCCGAGGCAGGGGGATCACGAGGTCAGGAGATCGAGACCATCCTGGCTAACACGGTGAAACCCCGTTTCTACTAAAAATACAAAAAAATTAGCCGGGCATGGTGGCAGGCTCCTGTAGTCCCAGCTACTCAGGAGGCTGAGGCAGGAGAATGGAGTGAACCCGGGAGGCGGAGCTTGCAGTGAGCCGAGATCACGCCATTGCACTGTAGCCTGGGCAACAGAGCCAGACTCTGTCTCAAAAAAAAAAAAAAAAAAATTCCCAAATTGAGAGAAGCAGATACCGTCTAGGCCCACAAAGGACACTTTTACCTATCCAGATGCAGATGTCTAATTTCTAAGGCTGTTTTTCCTAGGTAATCAGGAACATGGTTGGGGCCAGCAGTGGTGGGGCTGGACAGAGAGAGAAACTGAGACTCACCTCTGGCCAAAAAAGGGTCTGGCACCTGCTTAGGAGGGCTTCCAAAACTTTTTCAGCCTGTGGAAACAAACCCACAAGCAATGTGTTACTGGTCAGGGAACTAAAATCTGTTACCTAAATGCCAGGGGTTTATTCTAGGTTCTGCTGCTTGCAGCGCAGAAAGCCGGTCACTGAGACAGTGAAGATAGCCAGAGAAGGCTTTAATCAGGTGCTACAGCCAAGGAGATGACAGATAAGTTTCAAATCCATCTCCCCAAGCAACCAAAATTAGGGGTTTATATAGTAGGAAAGAAATGTACCCATGTATAGGAAAATAGAAACTAGGGAGGGGCAAGAAAGAGAAGTTGGTCAACAGGAAGCAGGTGGTTGGTTACGCAATCATGATGGGTGAGGGGGTCTTATGTTTCATTGTCAAGATGCAGTGATCTGCTAAGTTTCAGCTCCTTGATACTATCTGGAAGGCCTGATGGTTGTTTTCCTGAGAAAGGAACTCAGATAAGACAAACATAACTTTCTTGAGTTTTAAGACTGGAGCATCAATTTCTATGTTTATCCCAAAGAAACCATACACGTTAGTTCTATGAGACAACTGGGACAATTTCATAACCTGCATTAGAGGTTGGGAAATGTCTTCCCATATTTCCAAGATAAGAAAATGGTGCGGGTGCCAAAAGGCATTTGTCCCAGCCATACTCCCCTGTAATGTACACTTTTCTCAAACATTTTCCAAATTCACTGCTACAATGCTCTGAACCCTATTTTCTGCAGATGCAACTTTAACACTGTATATTCCCGATCTGATATGAAACAAACAACAGCACCACATTTTCCCTGGATATTGTTTTGCCAAGGCTTTGATCGCCACTGCAATGTGCCAGTAAGGCAAGAGGAAAATGAGAGATCTTTGAGTTCAATGATCTTGTCTAATTAAGCTGGAGCTCTTCCTTGGAACCCCAAGCCATCATGTTTGTGGTCAGCATTGGAGCTGTGAACATCCTTGCAACTGGTGTTTGGCACAACCTACTTTGATCAGCTTCTGTTCATCCACCAGCATTCCTATGCCATAGAGAGTCAATGAGGTAAGACTAAGCCAGATGGACCACCTACAAATCACCTGGAGACCTCTGAATCCATAGCAAAGAAATTAAGACAAAGTTGAGGAAATCTGAACTGCTGGTTCTCCAGCTACTGTAGTCAAATGAAGAGGACACAGAAGCCCTGAATTTTTGAATTCTTAGGATGAAAAGGGACACATTGGTTGTATGACTAACTTTATAGGGAAAGACCTTCAATACTTATAATGTGAACTCAATAAATAGTAATGTAGCAAAGAGAAGTTTCCATGGGGAAAATAACATGTAAAGTCAAAACTTAAAAAAAAAAAAAAGTAAGATTTTTCCAAGTTGCATTTCAGAAGAGGGAGGAGAAAAAGCAAAGGAAC")

_SEQUENCES = {
    "polya": _POLYA,
    "gc_rich": _GC_RICH,
    "cyp11a1": _CYP11A1,
    "gene_desert": _GENE_DESERT,
}

# Provenance for sequences fetched from hg38.
_SEQUENCE_PROVENANCE = {
    "polya": {
        "description": "All-adenine synthetic control (SEQ_LEN A's)",
    },
    "gc_rich": {
        "description": "Alternating GC synthetic control",
    },
    "cyp11a1": {
        "description": (
            "hg38 2114-bp window centered on CYP11A1 TSS "
            "(Ensembl ENSG00000140459, minus strand, TSS=74367884). "
            "Enformer-style centering: start=74366827, end=74368940 (1-based inclusive). "
            "Fetched from UCSC REST API 2026-08-21."
        ),
        "genome": "hg38",
        "chrom": "chr15",
        "center_1based": 74367884,
        "ucsc_0based_start": 74366826,
        "ucsc_0based_end": 74368940,
    },
    "gene_desert": {
        "description": (
            "hg38 2114-bp window centered on gene desert at chr8:127150000. "
            "Same locus as Enformer/Borzoi integration tests. "
            "Enformer-style centering: start=127148943, end=127151056 (1-based inclusive). "
            "Fetched from UCSC REST API 2026-08-21."
        ),
        "genome": "hg38",
        "chrom": "chr8",
        "center_1based": 127150000,
        "ucsc_0based_start": 127148943,
        "ucsc_0based_end": 127151057,
    },
}


def _seq_to_onehot(seq: str) -> np.ndarray:
    """Keras layout: (1, seq_len, 4)."""
    vocab = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((1, len(seq), 4), dtype=np.float32)
    for i, c in enumerate(seq.upper()):
        arr[0, i, vocab[c]] = 1.0
    return arr


def _validate_sequences():
    for name, seq in _SEQUENCES.items():
        if len(seq) != SEQ_LEN:
            raise ValueError(f"Sequence '{name}' has length {len(seq)}, expected {SEQ_LEN}")
        bad = set(seq) - set("ACGT")
        if bad:
            raise ValueError(f"Sequence '{name}' has non-ACGT chars: {bad}")


def main():
    parser = argparse.ArgumentParser(description="Generate ChromBPNet Keras goldens")
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to Adrenal_c0__fold_0__chrombpnet_nobias.h5 (Keras format)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON path (e.g. nucleobench/models/chrombpnet/testdata/adrenal_c0_keras_goldens.json)",
    )
    parser.add_argument(
        "--also-pytorch",
        action="store_true",
        help="Also score via bpnet-lite and print max |keras - pytorch|. Requires bpnet-lite in env.",
    )
    args = parser.parse_args()

    _validate_sequences()

    import tensorflow as tf  # noqa: PLC0415
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Loading weights from: {args.weights}")
    model = tf.keras.models.load_model(args.weights, compile=False)
    print("Model loaded.")

    examples = []
    for name, seq in _SEQUENCES.items():
        x = _seq_to_onehot(seq)
        outputs = model(x, training=False)
        # chrombpnet_nobias outputs: [profile (1, 1000), counts (1, 1)]
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
            keras_counts = float(outputs[1][0, 0])
        else:
            # Single-output fallback (should not happen for nobias).
            keras_counts = float(outputs[0, 0])
        print(f"  {name}: keras_counts={keras_counts:.6f}")
        examples.append(
            {
                "name": name,
                "sequence": seq,
                "sequence_provenance": _SEQUENCE_PROVENANCE[name],
                "keras_counts": keras_counts,
            }
        )

    if args.also_pytorch:
        import torch  # noqa: PLC0415
        from bpnetlite import BPNet  # noqa: PLC0415
        from nucleobench.models.bpnet.load_model import CountWrapper  # noqa: PLC0415

        print("\nbpnet-lite (PyTorch) scores for comparison:")
        pt_model = CountWrapper(BPNet.from_chrombpnet(args.weights))
        pt_model.eval()
        max_diff = 0.0
        for ex in examples:
            seq = ex["sequence"]
            vocab = {"A": 0, "C": 1, "G": 2, "T": 3}
            arr = np.array([[vocab[c] for c in seq.upper()]], dtype=np.int64)
            import torch.nn.functional as F  # noqa: PLC0415
            t = torch.tensor(arr, dtype=torch.long)
            one_hot = F.one_hot(t, num_classes=4).permute(0, 2, 1).float()
            with torch.no_grad():
                counts_pt = pt_model(one_hot).item()
            diff = abs(ex["keras_counts"] - counts_pt)
            max_diff = max(max_diff, diff)
            print(f"  {ex['name']}: keras={ex['keras_counts']:.6f}  pytorch={counts_pt:.6f}  |diff|={diff:.2e}")
        print(f"\nMax |keras - pytorch| across all sequences: {max_diff:.2e}")
        print("Use this to set atol in pytest.approx in the integration test.")

    output = {
        "provenance": {
            "model": "ChromBPNet",
            "cell_type": "Adrenal_c0",
            "fold": 0,
            "nobias": True,
            "zenodo_record": "https://zenodo.org/records/15048278",
            "nobias_relpath": "Adrenal_c0/Adrenal_c0__fold_0__chrombpnet_nobias.h5",
            "keras_load": "tf.keras.models.load_model(path, compile=False)",
            "counts_index": "outputs[1][0, 0]",
            "input_shape": "(1, 2114, 4)",
            "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "tensorflow_version": tf.__version__,
            "notes": (
                "keras_counts is the raw counts head scalar (log scale, NOT negated). "
                "NucleoBench oracle negates this: score = -keras_counts. "
                "The integration test checks: abs(-oracle(seq) - keras_counts) < atol."
            ),
        },
        "examples": examples,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"\nWrote {len(examples)} examples to: {args.out}")


if __name__ == "__main__":
    main()
