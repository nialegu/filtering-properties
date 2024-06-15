import numpy as np

def getData():
    w = 10
    start = 0
    end = 4
    step = 0.02
    x_values = [np.sin(w*y) for y in np.arange(start, end, step)] 
    data = np.asarray(x_values, dtype=float)
    return data
