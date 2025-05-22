import numpy as np
from sklearn.neighbors import NearestNeighbors


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


class KNNClassifier:
    def __init__(self, k=2, strategy='kd_tree', metric='euclidean', weights='False', test_block_size=100):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        if strategy != 'my_own':
            self.model = NearestNeighbors(n_neighbors=k, algorithm=strategy, metric=metric)
        else:
            pass

    def fit(self, X, y):
        self.y_train = y
        if self.strategy != 'my_own':
            self.model.fit(X)
        else:
            self.X_train = X

    def find_kneighbors(self, X, return_distance=False):
        num_samples = X.shape[0]
        indices = np.array([], dtype=np.int64).reshape((0, self.k))
        dists = np.array([], dtype=np.float64).reshape((0, self.k))
        for start_idx in range(0, num_samples, self.test_block_size):
            end_idx = min(start_idx + self.test_block_size, num_samples)
            X_block = X[start_idx:end_idx]
            if self.strategy != 'my_own':
                block_dists, block_indices = self.model.kneighbors(X_block)
            else:
                if self.metric == 'euclidean':
                    distan = euclidean_distance(X_block, self.X_train)
                else:
                    distan = cosine_distance(X_block, self.X_train)
                block_indices = np.argsort(distan, axis=1)[:, :self.k]
                if return_distance:
                    block_dists = np.take_along_axis(distan, block_indices, axis=1)
            indices = np.vstack((indices, block_indices))
            if return_distance:
                dists = np.vstack((dists, block_dists))
        if return_distance:
            return dists, indices
        else:
            return indices

    def predict(self, X):
        if self.weights is True:
            dists, indices = self.find_kneighbors(X, True)
            weight = np.array([1.0 / (x + 0.00001) for x in dists])
            answer = []
            for i in range(len(indices)):
                answer.append(np.argmax(np.bincount(self.y_train[indices[i]], weights=weight[i])))
            return answer
        else:
            answer = []
            indices = self.find_kneighbors(X, False)
            for i in range(len(indices)):
                answer.append(np.argmax(np.bincount(self.y_train[indices[i]])))
            return answer
