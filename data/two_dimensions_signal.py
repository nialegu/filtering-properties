import numpy as np

n_iterations = 10000
y_0 = 0.123
y_1 = 0.456

def f(a, b):
    y = np.zeros(n_iterations)
    y[0] = y_0
    y[1] = y_1
    for n in range(2, n_iterations):
        z = a*y[n-1] + b*y[n-2]
        while(abs(z) > 1):
            if (z > 1): z -= 2
            else: z += 2
        y[n] = z
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