import os
from pathlib import Path

import numpy as np

from .reporter_constants import BGH_3UTR, EGFP_CDS


class Saluki:
    """
    Saluki model for mRNA half-life prediction.
    Wrapper around the exact Keras model used in the paper.
    """

    def __init__(self, weights_path: str | Path | None = None):
        self.seq_length = 12288
        self.in_channels = 6

        # Weights location can be specified or falls back to default
        if weights_path is None:
            self.weights_path = Path(__file__).parent / "model0_best.h5"
        else:
            self.weights_path = Path(weights_path)
        self._load_model()

    def _load_model(self):  # pragma: no cover
        """Load the Keras model from the .h5 file."""
        # Force TensorFlow to use Keras 2 (tf-keras)
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        import tensorflow as tf

        class StochasticShift(tf.keras.layers.Layer):
            """
            Custom StochasticShift layer to handle loading the Saluki model.
            During inference, it acts as a pass-through (no shift).
            """

            def __init__(self, shift_max=0, symmetric=False, pad="uniform", **kwargs):
                super(StochasticShift, self).__init__(**kwargs)
                self.shift_max = shift_max
                self.symmetric = symmetric
                self.pad = pad

            def call(self, inputs, training=None):
                # Always return inputs as-is for inference/validation
                return inputs

            def get_config(self):
                config = super(StochasticShift, self).get_config()
                config.update(
                    {
                        "shift_max": self.shift_max,
                        "symmetric": self.symmetric,
                        "pad": self.pad,
                    }
                )
                return config

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Saluki weights not found at {self.weights_path}.")

        try:
            self.model = tf.keras.models.load_model(
                str(self.weights_path),
                custom_objects={"StochasticShift": StochasticShift},
                compile=False,
            )
        except Exception as e:
            print(f"Failed to load model from {self.weights_path}: {e}")
            raise e

    def predict_5utr(self, five_prime_utrs: list[str]) -> np.ndarray:
        """
        Predict mRNA half-life for a list of 5' UTR sequences by embedding them
        into a standard reporter construct (EGFP CDS + BGH 3' UTR).

        This mimics the experimental setup used for UTR evaluations.

        Args:
            five_prime_utrs: List of 5' UTR sequences.

        Returns:
            np.ndarray: Predicted half-lives.
        """
        sequences = []
        cds_starts = []
        cds_ends = []
        exon_ends_list = []

        for utr in five_prime_utrs:
            # Construct full mRNA: 5'UTR + EGFP + 3'UTR
            full_seq = utr + EGFP_CDS + BGH_3UTR

            # Calculate metadata based on the construction
            # CDS starts immediately after the UTR
            start = len(utr)
            # CDS ends after EGFP
            end = start + len(EGFP_CDS)

            sequences.append(full_seq)
            cds_starts.append(start)
            cds_ends.append(end)
            exon_ends_list.append([])  # No internal splicing in reporter

        return self.predict(sequences, cds_starts, cds_ends, exon_ends_list)

    def predict(
        self,
        sequences: list[str],
        cds_starts: list[int],
        cds_ends: list[int],
        exon_ends: list[list[int]],
    ) -> np.ndarray:
        """
        Predict mRNA half-life from sequence and annotations.

        Args:
            sequences: List of mRNA sequences.
            cds_starts: List of CDS start positions.
            cds_ends: List of CDS end positions.
            exon_ends: List of lists containing exon end positions.

        Returns:
            np.ndarray: Predicted half-lives.
        """
        encoded = self._encode_batch(sequences, cds_starts, cds_ends, exon_ends)

        # Ensure left-alignment (data at start, zeros at end)
        # This is critical for the backward GRU to see the data last.
        for i in range(encoded.shape[0]):
            non_zero = np.any(encoded[i] != 0, axis=-1)
            indices = np.where(non_zero)[0]
            if len(indices) > 0 and indices[0] > 0:
                shift = indices[0]
                encoded[i] = np.roll(encoded[i], shift=-shift, axis=0)

        return self.model.predict(encoded, verbose=0)

    def _encode_batch(self, sequences, cds_starts, cds_ends, exon_ends) -> np.ndarray:
        n = len(sequences)
        # Keras format: (Batch, Length, Channels)
        one_hot = np.zeros((n, self.seq_length, 6), dtype=np.float32)
        nt_map = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}

        for i in range(n):
            seq = sequences[i][: self.seq_length]
            for j, nt in enumerate(seq):
                if nt.upper() in nt_map:
                    one_hot[i, j, nt_map[nt.upper()]] = 1.0

            start = cds_starts[i]
            end_pos = cds_ends[i]
            if start >= 0 and end_pos >= start:
                # The paper says 1 for every 3rd position in the CDS.
                # We stop at the provided end_pos.
                for j in range(start, min(len(seq), end_pos + 1), 3):
                    one_hot[i, j, 4] = 1.0

            for end in exon_ends[i]:
                if end < self.seq_length:
                    one_hot[i, end, 5] = 1.0

        return one_hot

    @staticmethod
    def decode_one_hot(x: np.ndarray):
        """
        Decodes a single one-hot encoded sample back into metadata.

        Args:
            x: np.ndarray of shape (seq_length, 6)

        Returns:
            Tuple: (sequence string, cds_start, cds_end, exon_ends list)
        """
        nts = "ACGT"
        seq_chars = []
        for pos in range(x.shape[0]):
            idx = np.where(x[pos, :4] == 1)[0]
            # Use 'N' for positions with no nucleotide (padding)
            seq_chars.append(nts[idx[0]] if len(idx) > 0 else "N")
        sequence = "".join(seq_chars)

        cds_indices = np.where(x[:, 4] == 1)[0]
        if len(cds_indices) > 0:
            cds_start = int(cds_indices[0])
            cds_end = int(cds_indices[-1])
        else:
            cds_start = -1
            cds_end = -1

        exon_ends = np.where(x[:, 5] == 1)[0].astype(int).tolist()

        return sequence, cds_start, cds_end, exon_ends


if __name__ == "__main__":
    # Example usage
    pass
