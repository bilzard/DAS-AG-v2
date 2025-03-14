import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, use_deterministic_algorithm: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(use_deterministic_algorithm)


def is_deterministic_algorithm_enabled():
    return (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG", "") == ":4096:8"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG", "") == ":16:8"
    )
