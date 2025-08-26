import torch
import numpy as np
import os

def set_random_seeds(seed_value=42):
    """Set random seeds for reproducibility across various libraries and
    frameworks.

    This function sets the seed for PyTorch (both CPU and GPU), NumPy, and
    configures PyTorch's CUDA backend to ensure deterministic behavior.
    By setting these seeds, the experiments become reproducible.

    Parameters:
        seed_value (int, optional): The seed value to use for random number
                                    generation (default is 42).

    Effects:
        - Sets the seed for PyTorch's CPU and GPU operations.
        - Sets the seed for NumPy's random number generation.
        - Configures PyTorch to use deterministic algorithms and disables
          CUDA's benchmark feature for consistent behavior across runs.
    """
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    #fix seed


def GeneratorSeed(seed_value = 42):
    g = torch.Generator()
    g.manual_seed(seed_value)
    return g