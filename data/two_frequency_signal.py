import numpy as np

def compute_x(alpha, beta, w1, w2, y_values):
    x_values = alpha * np.sin(w1 * y_values) + beta * np.sin(w2 * y_values)
    return x_values

def getData():
    alpha = 0.7
    beta = 0.3
    if (alpha + beta != 1): 
        print("Coefficients are wrong")
    w1 = 1.0
    w2 = 2.0
    y_values = np.linspace(0, 10 * np.pi, 100)
    data = compute_x(alpha, beta, w1, w2, y_values)
    return data