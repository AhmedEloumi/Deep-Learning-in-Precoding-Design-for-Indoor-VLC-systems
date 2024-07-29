import numpy as np

def R_func(t, m):
    return (m + 1) * np.cos(t) ** m / (2 * np.pi)