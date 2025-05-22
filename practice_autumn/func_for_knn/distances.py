import numpy as np


def euclidean_distance(X, Y):
    X_s = np.sum(X ** 2, axis=1)[:, np.newaxis]
    Y_s = np.sum(Y ** 2, axis=1)[np.newaxis, :]
    t = X_s + Y_s - 2 * np.dot(X, Y.T)
    t = np.maximum(t, 0)
    return np.sqrt(t)


def cosine_distance(X, Y):
    a = np.linalg.norm(X, axis=1)[:, np.newaxis]
    b = np.linalg.norm(Y, axis=1)[np.newaxis, :]
    a[a == 0] = 1
    b[b == 0] = 1
    return 1 - (np.dot(X, Y.T) / (a * b))
