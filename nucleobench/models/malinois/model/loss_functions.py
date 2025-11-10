"""
MIT License

Copyright (c) 2025 Sagar Gosai, Rodrigo Castro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torch.nn as nn

from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, CTCLoss, NLLLoss, PoissonNLLLoss, GaussianNLLLoss, KLDivLoss, BCELoss, BCEWithLogitsLoss, MarginRankingLoss, HingeEmbeddingLoss, MultiLabelMarginLoss, HuberLoss, SmoothL1Loss, SoftMarginLoss, MultiLabelSoftMarginLoss, CosineEmbeddingLoss, MultiMarginLoss, TripletMarginLoss, TripletMarginWithDistanceLoss

##################
# Loss functions #
##################

class MSEKLmixed(nn.Module):
    """
    A custom loss module that combines Mean Squared Error (MSE) loss with Kullback-Leibler (KL) divergence loss.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
        alpha (float, optional): Scaling factor for the MSE loss term. Default is 1.0.
        beta (float, optional): Scaling factor for the KL divergence loss term. Default is 1.0.

    Attributes:
        reduction (str): The reduction method applied to the losses.
        alpha (float): Scaling factor for the MSE loss term.
        beta (float): Scaling factor for the KL divergence loss term.
        MSE (nn.MSELoss): The Mean Squared Error loss function.
        KL (nn.KLDivLoss): The Kullback-Leibler divergence loss function.

    Methods:
        forward(preds, targets):
            Calculate the combined loss by combining MSE and KL divergence losses.

    Example:
        loss_fn = MSEKLmixed()
        loss = loss_fn(predictions, targets)
    """
    
    def __init__(self, reduction='mean', alpha=1.0, beta=1.0):
        """
        Initialize the MSEKLmixed loss module.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
            alpha (float, optional): Scaling factor for the MSE loss term. Default is 1.0.
            beta (float, optional): Scaling factor for the KL divergence loss term. Default is 1.0.

        Returns:
            None
        """
        super().__init__()
        
        self.reduction = reduction
        self.alpha = alpha
        self.beta  = beta
        
        self.MSE = nn.MSELoss(reduction=reduction.replace('batch',''))
        self.KL  = nn.KLDivLoss(reduction=reduction, log_target=True)
        
    def forward(self, preds, targets):
        """
        Calculate the combined loss by combining MSE and KL divergence losses.

        Args:
            preds (Tensor): The predicted tensor.
            targets (Tensor): The target tensor.

        Returns:
            Tensor: The combined loss tensor.
        """
        preds_log_prob  = preds   - torch.logsumexp(preds, dim=-1, keepdim=True)
        target_log_prob = targets - torch.logsumexp(targets, dim=-1, keepdim=True)
        
        MSE_loss = self.MSE(preds, targets)
        KL_loss  = self.KL(preds_log_prob, target_log_prob)
        
        combined_loss = MSE_loss.mul(self.alpha) + \
                        KL_loss.mul(self.beta)
        
        return combined_loss.div(self.alpha+self.beta)

class L1KLmixed(nn.Module):
    """
    A custom loss module that combines L1 loss with Kullback-Leibler (KL) divergence loss.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
        alpha (float, optional): Scaling factor for the L1 loss term. Default is 1.0.
        beta (float, optional): Scaling factor for the KL divergence loss term. Default is 1.0.

    Attributes:
        reduction (str): The reduction method applied to the losses.
        alpha (float): Scaling factor for the L1 loss term.
        beta (float): Scaling factor for the KL divergence loss term.
        MSE (nn.L1Loss): The L1 loss function.
        KL (nn.KLDivLoss): The Kullback-Leibler divergence loss function.

    Methods:
        forward(preds, targets):
            Calculate the combined loss by combining L1 and KL divergence losses.

    Example:
        loss_fn = L1KLmixed()
        loss = loss_fn(predictions, targets)
    """
    
    def __init__(self, reduction='mean', alpha=1.0, beta=1.0):
        """
        Initialize the L1KLmixed loss module.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
            alpha (float, optional): Scaling factor for the L1 loss term. Default is 1.0.
            beta (float, optional): Scaling factor for the KL divergence loss term. Default is 1.0.

        Returns:
            None
        """
        super().__init__()
        
        self.reduction = reduction
        self.alpha = alpha
        self.beta  = beta
        
        self.MSE = nn.L1Loss(reduction=reduction.replace('batch',''))
        self.KL  = nn.KLDivLoss(reduction=reduction, log_target=True)
        
    def forward(self, preds, targets):
        """
        Calculate the combined loss by combining L1 and KL divergence losses.

        Args:
            preds (Tensor): The predicted tensor.
            targets (Tensor): The target tensor.

        Returns:
            Tensor: The combined loss tensor.
        """
        preds_log_prob  = preds   - torch.logsumexp(preds, dim=-1, keepdim=True)
        target_log_prob = targets - torch.logsumexp(targets, dim=-1, keepdim=True)
        
        MSE_loss = self.MSE(preds, targets)
        KL_loss  = self.KL(preds_log_prob, target_log_prob)
        
        combined_loss = MSE_loss.mul(self.alpha) + \
                        KL_loss.mul(self.beta)
        
        return combined_loss.div(self.alpha+self.beta)


L1Loss, MSELoss, CrossEntropyLoss, CTCLoss, NLLLoss, PoissonNLLLoss, GaussianNLLLoss, KLDivLoss, BCELoss, BCEWithLogitsLoss, MarginRankingLoss, HingeEmbeddingLoss, MultiLabelMarginLoss, HuberLoss, SmoothL1Loss, SoftMarginLoss, MultiLabelSoftMarginLoss, CosineEmbeddingLoss, MultiMarginLoss, TripletMarginLoss, TripletMarginWithDistanceLoss