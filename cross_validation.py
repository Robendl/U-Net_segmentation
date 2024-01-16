import numpy as np
from sklearn.model_selection import StratifiedKFold


def cross_validation(features, labels):
    X=features
    y=labels
    n_splits=4
    kf = StratifiedKFold(n_splits=n_splits)
    splits = []

    for train_index, test_index in kf.split(X, y):
        train_index = train_index.tolist()
        test_index = test_index.tolist()
        dataloader_trainset, dataloader_valset = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        splits.append((dataloader_trainset, dataloader_valset, y_train, y_test))

    return splits    