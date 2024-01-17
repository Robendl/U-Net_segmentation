import numpy as np
from sklearn.model_selection import train_test_split



def cross_validation(validation, features, labels):
    if validation == "8020":
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        splits = []
        splits.append((X_train, X_test, y_train, y_test))
    elif validation == "k_3fold":
        splits = k_fold_validation(X=features, y=labels, n_splits=3)
    elif validation == "k_5fold":
        splits = k_fold_validation(X=features, y=labels, n_splits=5)
    elif validation == "k_7fold":
        splits = k_fold_validation(X=features, y=labels, n_splits=7)
    else:
        raise Exception("Validation method not imported")    
    
    return splits