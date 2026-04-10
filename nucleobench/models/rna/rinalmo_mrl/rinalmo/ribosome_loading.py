"""Code from RinAlmo:
https://github.com/lbcb-sci/RiNALMo/blob/main/train_ribosome_loading.py
"""

import lightning.pytorch as pl

from .config import model_config
from .model.model import RiNALMo
from .model.downstream import RibosomeLoadingPredictionHead
from .utils.scaler import StandardScaler

class RibosomeLoadingPredictionWrapper(pl.LightningModule):
    def __init__(
        self,
        lm_config: str = "giga",
        head_embed_dim: int = 32,
        head_num_blocks: int = 6,
        force_cpu: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.scaler = StandardScaler()

        self.lm = RiNALMo(model_config(lm_config, force_cpu=force_cpu))

        self.pred_head = RibosomeLoadingPredictionHead(
            c_in=self.lm.config['model']['transformer'].embed_dim,
            embed_dim=head_embed_dim,
            num_blocks=head_num_blocks
        )

        self.pad_idx = self.lm.config['model']['embedding'].padding_idx

    def forward(self, tokens):
        x = self.lm(tokens)["representation"]

        # Nullify padding token representations
        pad_mask = tokens.eq(self.pad_idx)
        x[pad_mask, :] = 0.0

        pred = self.pred_head(x, pad_mask)
        pred = self.scaler.inverse_transform(pred).clamp(min=0.0) # "Unscale" predictions
        return pred