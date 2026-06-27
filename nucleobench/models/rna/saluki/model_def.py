import argparse

import numpy as np

from nucleobench.optimizations import model_class as mc

from . import load_model


class SalukiModel(mc.ModelClass):
    """Saluki model for mRNA half-life prediction."""

    @staticmethod
    def init_parser():
        """
        Add arguments to an argparse ArgumentParser.

        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.

        Returns:
            argparse.ArgumentParser: Argument parser with added arguments.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--override_weights_local_path", type=str, default=None)
        return parser

    @staticmethod
    def debug_init_args():
        return {}

    def __init__(
        self,
        override_weights_local_path: str | None = None,
        override_model=None,
    ):
        self.vocab = ["A", "C", "G", "T"]
        self.vocab_to_idx = {nt: i for i, nt in enumerate(self.vocab)}
        self.vocab_array = np.array(self.vocab)

        if override_model is not None:
            self.model = override_model
        else:
            self.model = load_model.load_model(weights_path=override_weights_local_path)

    def inference_on_strings(self, x: list[str]) -> np.ndarray:
        """
        Predict mRNA half-life for a list of 5' UTR sequences by embedding them
        into a standard reporter construct (EGFP CDS + BGH 3' UTR).
        """
        # Replace U with T for DNA processing
        seqs_dna = [seq.replace("U", "T") for seq in x]

        # If override_model is a mock callable, we can call it directly
        if hasattr(self.model, "predict_5utr"):
            preds = self.model.predict_5utr(seqs_dna)
        else:
            # If it's a mock/callable, call it
            preds = self.model(seqs_dna)

        # Ensure it returns a 1D numpy array
        preds = np.array(preds).flatten()

        # Multiply by -1 so "better" sequences (longer half-life, i.e. more stable)
        # are lower (minimized), according to convention.
        return -1 * preds

    def __call__(self, x: list[str]) -> np.ndarray:
        if isinstance(x, str):
            raise ValueError(f"Input needs to be list of strings, not just string: {x}")
        return self.inference_on_strings(x)
