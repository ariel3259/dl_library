import numpy as np

#activation function and its derivate
def tanth(x):
    return np.tanh(x)

def tanth_prime(x):
    return 1 - np.tanh(x)**2
