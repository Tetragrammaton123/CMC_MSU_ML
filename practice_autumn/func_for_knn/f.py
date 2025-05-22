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
            fold_score = accuracy_score(y_val, y_pred)
            scores_dict[k].append(fold_score)
    for k in scores_dict:
        scores_dict[k] = np.array(scores_dict[k])
    return scores_dict
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier