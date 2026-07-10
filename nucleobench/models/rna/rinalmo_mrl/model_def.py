"""Public interface for Rinalmo model.

To test on real data:
```zsh
python -m nucleobench.models.rna.rinalmo_mrl.model_def
```
"""

import argparse
from typing import Any

import numpy as np
import torch

from nucleobench.common import constants, string_utils
from nucleobench.optimizations import model_class as mc

from . import load_model
from .rinalmo.data.alphabet import Alphabet


class RinalmoMRL(mc.PyTorchDifferentiableModel, mc.TISMModelClass):
    """RinAlmo finetuned for mean ribosome loading prediction."""

    @staticmethod
    def init_parser():
        """
        Add arguments to an argparse ArgumentParser.

        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.

        Returns:
            argparse.ArgumentParser: Argument parser with added energy-specific arguments.

        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--override_ft_wts_local_path", type=str, default=None)
        return parser

    @staticmethod
    def debug_init_args():
        return {}

    def __init__(
        self,
        override_model: torch.nn.Module | None = None,
        override_ft_wts_local_path: str | None = None,
    ):
        self.has_cuda = torch.cuda.is_available()
        self.model: Any
        if override_model:
            self.model = override_model
        elif override_ft_wts_local_path:
            self.model = load_model.load_model(
                ft_wts_url=override_ft_wts_local_path, has_cuda=self.has_cuda
            )
        else:
            self.model = load_model.load_model(has_cuda=self.has_cuda)
        # Disable dropout for inference.
        self.model.eval()

        self.alphabet = Alphabet()

        self.vocab = constants.VOCAB
        self.vocab_to_idx = {nt: i for i, nt in enumerate(self.vocab)}
        self.vocab_array = np.array(self.vocab)

        # Cache nucleotide embeddings for efficient reuse in inference_on_tensor
        # RiNALMo token indices: A=5, C=6, G=7, T=8
        nucleotide_token_indices = torch.tensor([5, 6, 7, 8], dtype=torch.long)
        if self.has_cuda:
            nucleotide_token_indices = nucleotide_token_indices.cuda()
        self._nucleotide_embeddings = self.model.lm.embedding(
            nucleotide_token_indices
        ).detach()  # Shape: (4, embed_dim)

        # Cache CLS and EOS embeddings
        cls_eos_token_indices = torch.tensor([0, 2], dtype=torch.long)  # CLS=0, EOS=2
        if self.has_cuda:
            cls_eos_token_indices = cls_eos_token_indices.cuda()
        cls_eos_embeddings = self.model.lm.embedding(
            cls_eos_token_indices
        )  # Shape: (2, embed_dim)
        self._cls_embedding = cls_eos_embeddings[0:1].detach()  # Shape: (1, embed_dim)
        self._eos_embedding = cls_eos_embeddings[1:2].detach()  # Shape: (1, embed_dim)

    def _tokenize(self, x: str) -> torch.Tensor:
        encoded_seq = self.alphabet.encode(x)
        return torch.tensor(
            encoded_seq, dtype=torch.int64, device="gpu" if self.has_cuda else "cpu"
        )

    def batch_tokenize(self, x: list[str]) -> torch.Tensor:
        encoded_seq = self.alphabet.batch_tokenize(x)
        return torch.tensor(
            encoded_seq, dtype=torch.int64, device="gpu" if self.has_cuda else "cpu"
        )

    def _batch_embed(self, x: list[str]) -> torch.Tensor:
        return self.model.lm.embedding(self.batch_tokenize(x))

    def inference_on_tensor(self, x: torch.Tensor, return_debug_info: bool = False) -> torch.Tensor:
        """Run inference on a one-hot encoded tensor.

        IMPORTANT: This method ONLY accepts one-hot encoded tensors to ensure
        compatibility with gradient-based optimizations (Ledidi, FastSeqProp).
        Token indices are no longer supported as input.

        Args:
            x: One-hot encoded tensor of shape (batch_size, 4, seq_len)
               where dimension 1 corresponds to nucleotides [A, C, G, T/U]

        Returns:
            Tensor of shape (batch_size,) with model predictions
        """
        # ONLY accept one-hot encoded input (shape: batch_size, 4, seq_len)
        assert x.ndim == 3 and x.shape[1] == 4, (
            f"Expected one-hot tensor with shape (batch, 4, seq_len), got {x.shape}"
        )

        batch_size = x.shape[0]
        seq_len = x.shape[2]
        device = x.device

        # Use cached nucleotide embeddings (computed once during __init__)
        # Move to the same device as input if needed
        nucleotide_embeddings = self._nucleotide_embeddings.to(
            device
        )  # Shape: (4, embed_dim)

        # Compute weighted sum of embeddings using one-hot weights (soft indexing)
        # This maintains gradient flow through the one-hot weights
        x_transposed = x.transpose(1, 2)  # Shape: (batch_size, seq_len, 4)
        sequence_embeddings = torch.matmul(
            x_transposed, nucleotide_embeddings
        )  # Shape: (batch_size, seq_len, embed_dim)

        # Use cached CLS and EOS embeddings
        cls_embedding = self._cls_embedding.to(device).expand(
            batch_size, 1, -1
        )  # Shape: (batch_size, 1, embed_dim)
        eos_embedding = self._eos_embedding.to(device).expand(
            batch_size, 1, -1
        )  # Shape: (batch_size, 1, embed_dim)

        # Concatenate embeddings
        full_embeddings = torch.cat(
            [cls_embedding, sequence_embeddings, eos_embedding], dim=1
        )  # Shape: (batch_size, seq_len+2, embed_dim)

        # Create padding mask (all False since we're not padding)
        pad_mask = torch.zeros(batch_size, seq_len + 2, dtype=torch.bool, device=device)
        if self.has_cuda:
            pad_mask = pad_mask.cuda()

        # Forward through the transformer
        # Skip the embedding layer since we already have embeddings
        x_emb = full_embeddings

        # Forward through transformer
        if self.model.lm.config.model.transformer.use_flash_attn:
            key_padding_mask = torch.logical_not(pad_mask)
        else:
            key_padding_mask = pad_mask

        representation, _ = self.model.lm.transformer(
            x_emb,
            key_padding_mask=key_padding_mask,
            need_attn_weights=False,
        )

        # Apply prediction head
        # Nullify padding token representations (though we don't have any)
        representation[pad_mask, :] = 0.0

        ret = self.model.pred_head(representation, pad_mask)
        ret = self.model.scaler.inverse_transform(ret).clamp(min=0.0)

        assert isinstance(ret, torch.Tensor)
        assert ret.shape == (len(x),), ret.shape

        # Multiply by -1 so "better" sequences are lower, according to convention.
        return -1 * ret

    def inference_on_strings(self, x: list[str]) -> np.ndarray:
        # Convert strings to one-hot tensors
        # RiNALMo expects RNA sequences, but standard vocab uses T not U
        # So we convert U to T for one-hot encoding
        seqs_dna = [seq.replace("U", "T") for seq in x]
        batch_onehot = string_utils.dna2tensor_batch(seqs_dna, vocab_list=self.vocab)

        if self.has_cuda:
            batch_onehot = batch_onehot.cuda()

        ret = self.inference_on_tensor(batch_onehot)
        return ret.detach().clone().cpu().numpy()

    def __call__(self, x: list[str], return_debug_info: bool = False) -> np.ndarray:
        if isinstance(x, str):
            raise ValueError(f"Input needs to be list of strings, not just string: {x}")
        return self.inference_on_strings(x)
