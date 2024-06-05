import numpy as np

def get_sequence(x0, n_iterations):
    x = np.zeros(n_iterations)
    x[0] = x0
    for n in range(1, n_iterations):
        x[n] = 4 * x[n-1] * (1 - x[n-1])
    return x

def getData():
    # Initialize parameters
    initial_value = 0.1
    iterations = 100
    data = get_sequence(initial_value, iterations)
    return data