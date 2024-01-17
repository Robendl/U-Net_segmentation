import torch
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from network import UNet

class PyTorchWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, hyperparameter1, hyperparameter2, loss_function):
        # Initialize your wrapper with hyperparameters
        self.hyperparameter1 = hyperparameter1
        self.hyperparameter2 = hyperparameter2
        self.loss_function = loss_function

    def fit(self, X, y):
        # Instantiate and train your PyTorch model here using the provided hyperparameters
        self.model = UNet(self.hyperparameter1, self.hyperparameter2)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hyperparameter1)

        criterion = self.loss_function  # Use the provided loss function

        # ... training logic using X and y ...
        # Example training loop:
        for epoch in range(num_epochs):
            # Forward pass, backward pass, and optimization steps
            # ...

    def predict(self, X):
        # Make predictions using your trained model
        # ...

    def score(self, X, y):
        # Evaluate the performance of your model using a scoring metric (e.g., accuracy)
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

# Create a hyperparameter grid to search
param_grid = {
    'hyperparameter1': [value1, value2],
    'hyperparameter2': [value3, value4],
    'loss_function': [nn.CrossEntropyLoss(), nn.BCELoss()]  # Include different loss functions
}

# Create an instance of the PyTorch wrapper
pytorch_wrapper = PyTorchWrapper()

# Use GridSearchCV with your PyTorch wrapper
grid_search = GridSearchCV(pytorch_wrapper, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Access the best hyperparameters
best_hyperparameters = grid_search.best_params_

# Access the best trained model
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
test_score = best_model.score(X_test, y_test)
