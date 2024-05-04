from dawgs_ml.dataframe import dataframe as df
import utils


class Split:
    def __init__(self, data: df.DataFrame, target: str | int, seed=1):
        """Class to split data into train and test sets. If random seed parameter is not provided or is set to some
        non-zero value, then the dataframe is shuffled before splitting. If the random seed is set to 0, then no
        random shuffling is performed and the dataframe is sliced at the nth position(s) to create the other subsets."""
        self.data = data.copy()
        self.target = target
        self.random_seed = seed
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.X_val = None
        self.Y_val = None
        if self.random_seed != 0:
            self.data.shuffle(seed=self.random_seed)

    def train_test(self, test_size: float = 0.2) -> (df.DataFrame, df.Column, df.DataFrame, df.Column):
        """Method to split data into train and test sets.

        Args:
            test_size (float): proportion of the dataset to include in the test split. Default is 0.2

        Returns:
            values (tuple): 4 values in the following order:
            X_test, Y_test, X_train, Y_train
            """
        total_rows = self.data.rows
        test_size = int(total_rows * test_size)
        test_set = df.DataFrame()  # initialize empty test set dataframe
        test_set.set_columns_name(self.data.get_column_names())

        for i in range(test_size):
            test_set.add_row(self.data.pop_row(i))

        # After popping,the remaining rows in self.data is the train set
        self.Y_test = test_set.pop_column(self.target)
        self.X_test = test_set
        self.Y_train = self.data.pop_column(self.target)
        self.X_train = self.data

        return self.X_test, self.Y_test, self.X_train, self.Y_train

    def train_test_val(self, test_size=0.2, val_size=0.1):
        """Method to split data into train, test, and validation sets.

        Args:
            test_size (float): proportion of the dataset to include in the test split. Default is 0.2
            val_size (float): proportion of the dataset to include in the validation split. Default is 0.1

        Returns:
            values (tuple): 6 values in the following order:
            X_test, Y_test, X_train, Y_train, X_val, Y_val
            """
        total_rows = self.data.rows
        test_size = int(total_rows * test_size)
        val_size = int(total_rows * val_size)
        test_set = df.DataFrame()  # initialize empty test set dataframe
        val_set = df.DataFrame()  # initialize empty validation set dataframe
        test_set.set_columns_name(self.data.get_column_names())
        val_set.set_columns_name(self.data.get_column_names())

        for i in range(test_size):
            test_set.add_row(self.data.pop_row(i))

        for i in range(val_size):
            val_set.add_row(self.data.pop_row(i))

        # After popping, the remaining rows in self.data is the train set
        self.Y_test = test_set.pop_column(self.target)
        self.X_test = test_set
        self.Y_val = val_set.pop_column(self.target)
        self.X_val = val_set
        self.Y_train = self.data.pop_column(self.target)
        self.X_train = self.data
        return self.X_test, self.Y_test, self.X_train, self.Y_train, self.X_val, self.Y_val


class KFoldCrossValidation:
    def __init__(self, data: df.DataFrame, target_column: str | int, k: int, seed: int = 1):
        self._k = k
        self._data = data
        self._target_column = target_column
        self._s = seed
        self._folds = []
        if k <= 1 or k >= self._data.rows:
            raise ValueError("k must be greater than 1 and less than the number of rows in the dataframe")

    @property
    def folds(self) -> list[tuple[df.DataFrame, df.Column]]:
        return self._folds

    def split(self):
        """Method to split the dataframe into k folds. It creates an instance of the Split class, and it iteratively
        calls the train_test method to split the dataframe into k folds, by adjusting the ratio at every iteration so
        to make folds of the same size. The leftover is the last fold so that train_test split is called k-1 times.
        It returns a list of folds [(X1_test, Y1_test), (X2_test, Y2_test), ...]. """
        ratio = 1 / self._k
        split_size = int(self._data.rows / self._k)
        resizable_data = self._data
        for i in range(self._k - 1):
            split = Split(data=resizable_data, target=self._target_column, seed=self._s)
            X_test, Y_test, X_train, Y_train = split.train_test(test_size=ratio)
            self._folds.append((X_test, Y_test))
            resizable_data = X_train
            resizable_data.add_column(Y_train)
            ratio = split_size / self._data.rows  # update the ratio for split for next iteration
        last_fold_Y = resizable_data.pop_column(self._target_column)
        self._folds.append((resizable_data, last_fold_Y))


class Evaluation:
    def __init__(self, predictions, actual):
        self.predictions = predictions
        self.actual = actual
        self.tp, self.fp, self.fn, self.tn = self._tp_fp_fn_tn()

    def _tp_fp_fn_tn(self):
        tp, fp, fn, tn = (0,) * 4
        for i in range(len(self.predictions)):
            if self.predictions[i] == 1 and self.actual[i] == 1:
                tp += 1
            elif self.predictions[i] == 1 and self.actual[i] == 0:
                fp += 1
            elif self.predictions[i] == 0 and self.actual[i] == 1:
                fn += 1
            else:
                tn += 1
        return tp, fp, fn, tn

    def accuracy(self):
        """Returns the accuracy of the model on the test set, this metric for model evaluation is recommended for
        balanced classes."""
        correct = 0
        for i in range(len(self.predictions)):
            if self.predictions[i] == self.actual[i]:
                correct += 1
        return correct / len(self.predictions)

    def precision(self):
        """Returns the precision of the model on the test set, this metric for model evaluation is recommended for both
        imbalanced and balanced classes."""
        return self.tp / (self.tp + self.fp)

    def recall(self):
        """Returns the recall of the model on the test set, this metric for model evaluation is recommended for both
        imbalanced and balanced classes."""
        return self.tp / (self.tp + self.fn)

    def f1_score(self):
        """Returns the harmonic mean of the precision and recall of the model on the test set, this metric for model
        evaluation is recommended for both imbalanced and balanced classes."""
        return (2 * self.precision() * self.recall()) / (self.precision() + self.recall())

    def confusion_matrix(self) -> list[list[int]]:
        """Returns:
            list[list[int]]: confusion matrix of the model on the test set."""
        print("\t\t\tPredicted - | Predicted + |")
        print(f"Actual - |\t\t {self.tn} \t\t {self.fp}")
        print(f"Actual + |\t\t {self.fn} \t\t\t {self.tp}")
        return [[self.tn, self.fp], [self.fn, self.tp]]


class Combination:
    """Class to store the metrics of each combination."""

    def __init__(self, target_metric: str = "accuracy"):
        self.target_metric = target_metric
        self._fold = _Folds()
        self.combination = {}
        self.best_metric_value = float('-inf')

    def _define_best(self, folds: '_Folds', combination: dict):
        if self.target_metric not in folds.metrics:
            raise ValueError(f"{self.target_metric} is not a recognized metric.")

        # Calculate the new average metric value.
        if folds.metrics[self.target_metric]:
            total = sum(folds.metrics[self.target_metric])
            count = len(folds.metrics[self.target_metric])
            new_metric_value = total / count
        else:
            new_metric_value = float('-inf')  # default value if no metrics exists

        if new_metric_value > self.best_metric_value:   # update the best combination-metric if the new metric is better
            self.best_metric_value = new_metric_value
            self._fold = folds
            self.combination = combination

    def _get_best_combination(self):
        """Returns the best combination and its average metric value."""
        if not self.combination:
            return None, None  # Handle the case where no combination has been defined yet.
        return self.combination, self.best_metric_value

    def __repr__(self):
        return str(f"Best combination: {self.combination}, Average {self.target_metric}: {self.best_metric_value}")


class _Folds:
    """Helper class to store the metrics of each fold."""
    def __init__(self):
        self.metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }

    def update(self, accuracy, precision, recall, f1_score):
        self.metrics['accuracy'].append(accuracy)
        self.metrics['precision'].append(precision)
        self.metrics['recall'].append(recall)
        self.metrics['f1_score'].append(f1_score)

    def get_average(self, metric):
        if metric not in self.metrics:
            raise ValueError(f"Metric {metric} is not valid.")
        if not self.metrics[metric]:
            return None  # or raise an error if preferred
        return sum(self.metrics[metric]) / len(self.metrics[metric])

    def __repr__(self):
        return str(self.metrics)


class GridSearch:
    """Given a dataframe and a list of hyperparameters, this class will return the best possible combination of
    hyperparameters by running a grid search with k-fold cross validation."""

    def __init__(self, data: df.DataFrame, model_instance, k_folds: int = 5, **hyperparameters):
        self.data = data.copy()  # use the deep copy of the dataframe to avoid changing the original dataframe
        self.model_instance = model_instance
        self.k_folds = k_folds
        if self._correctness_model_params(hyperparameters):
            self.hyperparameters = hyperparameters
        else:
            raise AttributeError("Incorrect hyperparameters for the model instance.")

    def _correctness_model_params(self, hyperparameters):
        """Helper method to check if the hyperparameters of the model instance are correct."""
        for key, value in hyperparameters.items():
            if not hasattr(self.model_instance, key):
                return False
        return True

    def _set_model_params(self, parameters_combination: list):
        """Helper method to set the hyperparameters of the model instance."""
        for i in (range(len(parameters_combination))):
            setattr(self.model_instance, list(self.hyperparameters)[i], parameters_combination[i])

    def _create_train_val_fold(self, target_col):
        """Helper method to create train and validation folds for k-fold cross validation."""
        K = KFoldCrossValidation(data=self.data, target_column=target_col, k=self.k_folds)
        K.split()
        sets = []
        for fold in K.folds:
            temp_df, temp_target = df.DataFrame(), df.Column(target_col)
            validation = fold  # leave one fold out for evaluation on validation set
            for leftover_fold in K.folds:
                if leftover_fold != fold:
                    temp_df.merge(leftover_fold[0])
                    temp_target.name = leftover_fold[1].name
                    temp_target.append_column(leftover_fold[1])
            sets.append((temp_df, temp_target, validation[0], validation[1]))
        return sets

    def return_model_instance(self):
        """Return the model instance as a dictionary with the parameters and their values."""
        return self.model_instance.__dict__

    def run_training(self, target_col: str | int, metric: str = "accuracy"):
        """This method runs the grid search with k-fold cross validation, and it returns the best combination of
        hyperparameters based on the specified metric.

        Args:
            target_col: The name of the target column in the dataframe.
            metric: The metric to use for evaluating the best combination. Default is accuracy.
            """
        folds = self._create_train_val_fold(target_col)
        hyperparameters_combinations = utils.cartesian_product(*self.hyperparameters.values())
        best_combination = Combination(target_metric=metric)
        for combination in hyperparameters_combinations:
            print(f"Running K-fold cross validation with hyperparameters: {combination}")
            fold_metric = _Folds()
            for fold in folds:
                self._set_model_params(combination)
                print(f"Fold: {folds.index(fold) + 1}")
                self.model_instance.fit(fold[0], fold[1], fold[0], fold[1])
                validation_predictions = self.model_instance.predict(fold[2])
                metrics = Evaluation(validation_predictions, fold[3])
                fold_metric.update(metrics.accuracy(), metrics.precision(), metrics.recall(), metrics.f1_score())
                print(f"Accuracy updated to: {fold_metric.get_average(metric)}")
                self.model_instance.__init__()  # reset model parameters after each combination
            best_combination._define_best(fold_metric, combination)
        print(best_combination)
        return best_combination



class RandomSearch:
    """Given a range of continuous values hyperparameters, this class will random sample hyperparameters using the
    k-fold cross validation."""
    pass


class BayesianSearch:
    """Use bayesian optimization to optimize hyperparameters, by focusing more on the most promising hyperparameters"""
    pass
