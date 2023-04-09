import random
import numpy as np
import torch

# set the random seed

def set_seed(seed: int):
    # set the random seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
