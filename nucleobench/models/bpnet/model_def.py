"""BPNet models from the Ledidi paper (https://icml-compbio.github.io/icml-website-2020/2020/papers/WCBICML2020_paper_23.pdf).

The authors have trained a BPNet model for each of eight proteins in K562 
whose ChIP-seq data is on the ENCODE portal. Each model was trained on the union 
of reads from two replicate BAM files mapped at basepair resolution and separated 
out by strand. These models were trained using the bpnet-lite repository.

Stored at https://zenodo.org/records/14604495.

To test on real data:
```zsh
python -m nucleobench.models.bpnet.model_def
```
"""

from typing import Optional, Union

import argparse
import gc
import numpy as np
import torch

from nucleobench.common import string_utils
from nucleobench.common import attribution_lib_torch as att_lib

from nucleobench.optimizations import model_class as mc
from nucleobench.models.bpnet import load_model
from nucleobench.models.bpnet import constants as bp_constants


class BPNet(mc.PyTorchDifferentiableModel, mc.TISMModelClass):
    """BPNet model trained on eight proteins in K562 whose ChIP-seq 
    data is on the ENCODE portal. Each model was trained on the union 
    of reads from two replicate BAM files mapped at basepair resolution 
    and separated out by strand. These models were trained using the 
    bpnet-lite repository."""

    @staticmethod
    def init_parser():
        """
        Add energy-specific arguments to an argparse ArgumentParser.

        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.

        Returns:
            argparse.ArgumentParser: Argument parser with added energy-specific arguments.

        """
        parser = argparse.ArgumentParser()
        group = parser.add_argument_group("BPNet init args")
        group.add_argument("--protein", type=str, required=True, 
                           choices=bp_constants.AVAILABLE_MODELS_),
        
        return parser
    
    @staticmethod
    def debug_init_args():
        return {
            'protein': 'GATA2',
        }

    def __init__(
        self,
        protein: str,
        # The vocab MUST be this, since this is what was used to train the BPNets.
        vocab: list[str] = bp_constants.VOCAB_,
        override_model: Optional[torch.nn.Module] = None,
    ):
        self.protein = protein
        if override_model:
            self.model = override_model
        else:
            self.model = load_model.download(protein)

        # Consistent vocab is important for interpreting smoothgrad.
        self.vocab = vocab
        
        self.has_cuda = torch.cuda.is_available()


    def inference_on_tensor(
        self, 
        x: torch.Tensor,
        return_debug_info: bool = False,
        ) -> torch.Tensor:
        """Run inference on a one-hot tensor."""
        assert x.ndim == 3  # Batched.
        assert x.shape[1] == 4

        m_out = self.model(x)
        assert m_out.ndim == 2
        assert m_out.shape[1] == 1
        ret = torch.squeeze(m_out, dim=1)
        
        # Always return something that should be minimized, so flip the sign.
        ret *= -1
        
        return ret

    def inference_on_strings(self, x: list[str]) -> np.ndarray:
        tensor = string_utils.dna2tensor_batch(x, vocab_list=self.vocab)
        ret = self.inference_on_tensor(tensor)
        return ret.detach().clone().numpy()

    def __call__(self, x: list[str], return_debug_info: bool = False) -> np.ndarray:
        if isinstance(x, str):
            raise ValueError(f'Malinois input needs to be list of strings, not just string: {x}')
        ret = self.inference_on_strings(x)
        if return_debug_info:
            return ret, {}
        else:
            return ret


if __name__ == "__main__":
    # Test with a real model.
    import time
    for prot in bp_constants.AVAILABLE_MODELS_:
        print(f'Starting {prot}...')
        m = BPNet(protein=prot)
        ntimes = 100
        s_time = time.time()
        for _ in range(ntimes):
            m(["A" * 3_000])
        e_time = time.time()
        print(f'Finished {prot} in {e_time - s_time} seconds: {(e_time - s_time) / ntimes} s / iter')
