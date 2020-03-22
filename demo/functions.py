import numpy as np

def drift(t, x):
    return x * np.sin(t)


def diffusion(t, x):
    return x
