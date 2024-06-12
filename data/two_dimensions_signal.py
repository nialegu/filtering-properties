import numpy as np

n_iterations = 100
y_0 = -0.01
y_1 = 0.01

def f(a, b):
    y = np.zeros(n_iterations)
    y[0] = y_0
    y[1] = y_1
    for n in range(2, n_iterations):
        z = a*y[n-1] + b*y[n-2]
        if 1 < z < 3:
            y[n] = z - 2
        elif abs(z) <= 1:
            y[n] = z
        elif -3 < z < -1:
            y[n] = z + 2
        else: break
    return y

def getFirstData():
    a = 1
    b = 5
    return f(a, b)

def getSecondData():
    a = 5
    b = 3
    return f(a, b)

def getThirdData():
    a = -5
    b = 3
    return f(a, b)