"""Utilities for managing memory."""

import gc
import torch

def free_memory(debug: bool = False):
    collected = gc.collect()
    if debug:
        print(f'[free_memory] Collected: {collected}')
    torch.cuda.empty_cache()