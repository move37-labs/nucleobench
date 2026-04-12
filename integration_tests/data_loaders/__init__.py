"""Data loaders for integration tests."""

from .base import DataLoader
from .muscle_gene_expression_by_bucket import MuscleGeneExpressionByBucket

__all__ = ['DataLoader', 'MuscleGeneExpressionByBucket']