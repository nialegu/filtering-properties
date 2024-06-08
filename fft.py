import numpy as np

def DFT(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def FFT(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N <= 32:
        return DFT(x)
    elif N % 2 > 0:
        raise ValueError("The variable of x must be even")
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([
            X_even + factor[:N // 2] * X_odd,
            X_even + factor[N // 2:] * X_odd 
        ])
    
    