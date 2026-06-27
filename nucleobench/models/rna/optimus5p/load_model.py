import torch

from . import constants
from .model.optimus_5p import Optimus5Prime


def load_model(
    weights_path: str | None = None,
    has_cuda: bool | None = None,
) -> Optimus5Prime:
    if weights_path is None:
        weights_path = constants.CACHE_WEIGHTS_

    if has_cuda is None:
        has_cuda = torch.cuda.is_available()

    device = "cuda" if has_cuda else "cpu"

    model = Optimus5Prime(weights_path=weights_path)
    model.to(device)
    model.eval()
    return model
