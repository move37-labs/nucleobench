"""ChromBPNet oracle.

Two data sources are supported:

HDMA fetal-tissue models (Zenodo CC-BY 4.0):
  Paper: Liu*, Jessa*, Kim*, Ng* et al., bioRxiv 2025 (HDMA).
  Weights: https://zenodo.org/records/15048278

ENCODE K562 models (HuggingFace, ENCODE data-use policy):
  Collection: https://huggingface.co/collections/kundajelab/encode-chrombpnet-models
  Cite: Pampari et al. 2024.

Both are loaded via bpnet-lite `BPNet.from_chrombpnet` (PyTorch; no TensorFlow).

Scalar objective is the counts head, negated so NucleoBench optimizers
minimize (suppress predicted accessibility), matching BPNet ATAC.

Input sequences must be at least 2114 bp (ChromBPNet default receptive field).

To test on real data:
```zsh
python -m nucleobench.models.chrombpnet.model_def
```
"""

import argparse

import numpy as np
import torch

from nucleobench.common import string_utils
from nucleobench.models.chrombpnet import constants as cb_constants
from nucleobench.optimizations import model_class as mc


class ChromBPNetOracle(mc.PyTorchDifferentiableModel, mc.TISMModelClass):
    """Cell-type-specific ChromBPNet ATAC oracle.

    Supports HDMA fetal-tissue models (Zenodo) and ENCODE K562 models
    (HuggingFace).  See `constants.ALL_AVAILABLE_MODELS_` for the full list.
    """

    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser()
        group = parser.add_argument_group("ChromBPNetOracle init args")
        group.add_argument(
            "--cell_type",
            type=str,
            required=True,
            choices=cb_constants.ALL_AVAILABLE_MODELS_,
            help=(
                "HDMA cluster id (e.g. Adrenal_c0) or ENCODE K562 key "
                "(e.g. K562_ENCSR483RKN)."
            ),
        )
        return parser

    @staticmethod
    def debug_init_args():
        return {"cell_type": "Adrenal_c0"}

    def __init__(
        self,
        cell_type: str,
        vocab: list[str] = cb_constants.VOCAB_,
        override_model: torch.nn.Module | None = None,
        override_weights_local_path: str | None = None,
    ):
        self.cell_type = cell_type
        self._require_seq_len = override_model is None
        if override_model:
            self.model = override_model
        elif cell_type in cb_constants.K562_AVAILABLE_MODELS_:
            from nucleobench.models.chrombpnet import load_model_k562

            self.model = load_model_k562.download(
                cell_type,
                override_weights_local_path=override_weights_local_path,
            )
        else:
            from nucleobench.models.chrombpnet import load_model

            self.model = load_model.download(
                cell_type,
                override_weights_local_path=override_weights_local_path,
            )

        self.model.eval()

        self.vocab = vocab
        self.vocab_to_idx = {nt: i for i, nt in enumerate(vocab)}
        self.vocab_array = np.array(vocab)

    def inference_on_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference on a one-hot tensor of shape (batch, 4, seq_len)."""
        if x.ndim != 3 or x.shape[1] != 4:
            raise ValueError(f"Expected (batch, 4, seq_len), got {tuple(x.shape)}")
        if self._require_seq_len and x.shape[2] < cb_constants.SEQ_LEN:
            raise ValueError(
                f"Sequence length {x.shape[2]} < ChromBPNet SEQ_LEN="
                f"{cb_constants.SEQ_LEN}."
            )

        m_out = self.model(x)
        if m_out.ndim == 2 and m_out.shape[1] == 1:
            ret = torch.squeeze(m_out, dim=1)
        else:
            ret = m_out

        ret = ret * -1
        return ret

    def inference_on_strings(self, x: list[str]) -> np.ndarray:
        tensor = string_utils.dna2tensor_batch(x, vocab_list=self.vocab)
        ret = self.inference_on_tensor(tensor)
        return ret.detach().clone().numpy()

    def __call__(self, x: list[str]) -> np.ndarray:
        if isinstance(x, str):
            raise ValueError(
                f"Input must be a list of strings, not a single string: {x}"
            )
        return self.inference_on_strings(x)


if __name__ == "__main__":
    import time

    print("Loading ChromBPNetOracle(Adrenal_c0) ...")
    m = ChromBPNetOracle(cell_type="Adrenal_c0")
    seqs = [
        "A" * cb_constants.SEQ_LEN,
        ("GC" * (cb_constants.SEQ_LEN // 2))[: cb_constants.SEQ_LEN],
    ]
    labels = ["poly-A", "GC-rich"]
    for seq, label in zip(seqs, labels):
        t0 = time.perf_counter()
        score = m([seq])[0]
        ms = (time.perf_counter() - t0) * 1000
        print(f"  {label}: score={score:.4f}  ({ms:.0f} ms)")
