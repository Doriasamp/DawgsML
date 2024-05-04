from dataframe import dataframe as df  # this is my custom data structure for dataframes
from utils import utils
from utils.model_utils import Evaluation
import random
from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, random_seed: int = 1):
        self._X = None  # the actual data (features only), expecting an instance of DataFrame
        self._Y = None  # the target column, an instance of Column, or a python list
        self._X_val = None
        self._Y_val = None
        self.random_seed = random_seed

    @abstractmethod
    def _predict(self, x: list):
        pass

    @abstractmethod
    def _predictions(self):
        pass

    @abstractmethod
    def _validation_predictions(self):
        pass

    @abstractmethod
    def fit(self, train_data: df.DataFrame, train_target: df.Column | list, test_data: df.DataFrame,
            test_target: df.Column | list):
        pass

    @abstractmethod
    def predict(self, validation_x: df.DataFrame) -> list:
        pass


class Perceptron(Model):
    def __init__(self, learning_rate=0.1, epochs=10, activation: float = 0, verbosity: bool = False):
        super().__init__()
        self._weights = []
        self._step_activation = activation
        self.verbosity = verbosity
        self._bias = 0  # introduced bias for non-linearly separable problems having to pass through the origin at init
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._errors = []
        self._accuracy = 0
        self.train_accuracy = 0
        self.test_accuracy = 0
        random.seed(self.random_seed)

    @property
    def weights(self) -> list:
        return self._weights

    @weights.setter
    def weights(self, weights: list):
        self._weights = weights

    @property
    def X(self) -> df.DataFrame:
        return self._X

    @property
    def Y(self) -> list:
        return self._Y

    @property
    def X_val(self) -> df.DataFrame:
        return self._X_val

    @property
    def Y_val(self) -> list:
        return self._Y_val

    @property
    def step_activation(self):
        return self._step_activation

    @step_activation.setter
    def step_activation(self, step_activation):
        self._step_activation = step_activation

    def activation(self, value):
        return 1 if value > self._step_activation else 0

    def initialize_weights(self):
        for _ in range(self._X.columns):  # for each self.data.columns:
            weight = (random.random() - 0.5) * 0.1  # random value between -0.05 and 0.05
            self._weights.append(weight)

    def fit(self, train_data: df.DataFrame, train_target: df.Column | list, test_data: df.DataFrame,
            test_target: df.Column | list):
        """ Fit the model to the training data, accuracy is calculated on the validation data, but if you do not have
        a validation data, then use the training data again as validation data, especially in the context of just having
        the need to train a model on the whole dataset after running a Search algorithm for hyperparameters in k-fold.

        Args:
            train_data: The training data.
            train_target: The training target.
            test_data: The test data.
            test_target: The test target.
        """
        self._X = train_data
        self._Y = train_target
        self._X_val = test_data
        self._Y_val = test_target
        self._bias = 0
        self._weights = []
        self.initialize_weights()

        # fit the model
        for epoch in range(self.epochs):
            error = 0
            for i in range(self._X.rows):
                x_row = self._X[i]
                y = self._Y[i]
                y_prediction = self._predict(x_row)
                if y != y_prediction:
                    error += 1
                    for j in range(len(x_row)):  # update the weights
                        self._weights[j] += self.learning_rate * (y - y_prediction) * x_row[j]  # update rule
                    self._bias += self.learning_rate * (y - y_prediction)  # update the bias
            train_prediction = self._predictions()
            val_prediction = self._validation_predictions()
            train_accuracy = Evaluation(train_prediction, self._Y).accuracy()
            val_accuracy = Evaluation(val_prediction, self._Y_val).accuracy()
            if self.verbosity:
                print(f"Epoch: {epoch} || Error: {error}, Bias: {round(self._bias, 4)}, "
                      f"Train accuracy: {round(train_accuracy, 4)}, Validation accuracy: {round(val_accuracy, 4)}")

    def _predict(self, x):
        """Helper function that predicts the output for a given input and returns 0 or 1 based on activation step"""
        y = 0
        for i in range(len(x)):
            y += self._weights[i] * x[i]
        y += self._bias
        return self.activation(y)

    def _predictions(self):
        """Helper function that uses the trained model, after fitting, to make predictions on the train data."""
        predictions = []
        for i in range(self._X.rows):
            predictions.append(self._predict(self._X[i]))
        return predictions

    def _validation_predictions(self):
        """Helper function that uses the trained model, after fitting, to make predictions on the validation data."""
        predictions = []
        for i in range(self._X_val.rows):
            predictions.append(self._predict(self._X_val[i]))
        return predictions

    def predict(self, validation_x: df.DataFrame) -> list:
        """It uses the trained model, after fitting, to make predictions on the validation data."""
        predictions = []
        for i in range(validation_x.rows):
            value = utils.dot_product(validation_x[i], self._weights)
            value += self._bias
            predictions.append(self.activation(value))
        return predictions
