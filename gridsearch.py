import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import BaseEstimator, ClassifierMixin
from network import UnetWithHeader
from unet import bce_loss, dice_loss, combined_loss

from unet import train
from test import test


class PyTorchWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate, batch_size, loss_function):
        # Initialize your wrapper with hyperparameters
        self.model = UnetWithHeader(n_channels=3, n_classes=1, mode="mlp")
        self.model.cuda()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_function = loss_function

    def fit(self, X, y):
        num_epochs = 15
        train(self.model, X, self.learning_rate, self.batch_size, self.loss_function, num_epochs)

    def score(self, X, y, sample_weight=None):
        return test(self.model, X)


def gridsearch():
    # Create a hyperparameter grid to search
    param_grid = {
        'learning_rate': [0.001, 0.0001, 0.00001],
        'batch_size': [8],
        'loss_function': [bce_loss, dice_loss, combined_loss]
    }
    # param_grid = {
    #     'learning_rate': [0.001],
    #     'batch_size': [8],
    #     'loss_function': [bce_loss]
    # }

    # Create an instance of the PyTorch wrapper
    pytorch_wrapper = PyTorchWrapper(0.001, 24, bce_loss)

    # Use GridSearchCV with your PyTorch wrapper
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pytorch_wrapper, param_grid, cv=kf)

    total_images = 3064
    indices = np.arange(1, total_images + 1)
    np.random.shuffle(indices)
    data_used = 1000
    indices = indices[:data_used]
    train_size = int(data_used * 0.9)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    grid_search.fit(train_indices, train_indices)

    # Access the best hyperparameters
    best_hyperparameters = grid_search.best_params_
    print(best_hyperparameters)

    best_model = grid_search.best_estimator_

    test_score = best_model.score(test_indices, test_indices)
    print("Best test score:", test_score)


if __name__ == '__main__':
    gridsearch()
