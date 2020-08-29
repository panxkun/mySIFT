import numpy as np

def Gaussian_Kernel(sigma = 1, window_size = 3):

    window_half = int(window_size / 2)

    x = []
    y = []
    for row in np.linspace(-window_half, window_half, window_size):
        for col in np.linspace(-window_half, window_half, window_size):
            x.append(row)
            y.append(col)

    x = np.asarray(x)
    y = np.asarray(y)

    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

    sum = np.sum(g)

    g = g/sum

    kernel = np.zeros([window_size,window_size],dtype='float32')
    for row in range(window_size):
        for col in range(window_size):
            kernel[row][col] = g[row * window_size + col]

    return kernel