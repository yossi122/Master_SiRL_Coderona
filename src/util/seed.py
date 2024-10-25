import random
import numpy as np

CONSTANT_SEED = None  # None means 'do not fix a seed'

def set_random_seed():
    """
    Fixes the seed of all random number generators in the code to be
    CONSTANT_SEED. If it is None, does not fix seeds
    (instead, time is taken by default)
    :return:
    """
    if CONSTANT_SEED is not None:
        random.seed(CONSTANT_SEED)
        np.random.seed(CONSTANT_SEED)

