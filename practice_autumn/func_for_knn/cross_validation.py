import numpy as np
import nearest_neighbors

np.int = int


def kfold(n, n_folds):
    indices = np.arange(n)
    fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[:n % n_folds] += 1
    current = 0
    folds = []

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_indices, val_indices))
        current = stop

    return folds


def knn_cross_val_score(X, y, k_list, score='accuracy', cv=None, **kwargs):
    if cv is None:
        n = X.shape[0]
        n_folds = 5
        cv = kfold(n, n_folds)
    scores_dict = {k: [] for k in k_list}
    for k in k_list:
        knn = nearest_neighbors.KNNClassifier(k, **kwargs)
        for (train_indices, val_indices) in cv:
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_val)
            fold_score = calculate_accuracy(y_val, y_pred)
            scores_dict[k].append(fold_score)
    for k in scores_dict:
        scores_dict[k] = np.array(scores_dict[k])
    return scores_dict


def calculate_accuracy(pred, real):
    return np.mean(np.array(pred) == np.array(real))
