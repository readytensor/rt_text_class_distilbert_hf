import os
import warnings
import joblib
import numpy as np
import pandas as pd
from config import paths
from datasets import Dataset
from sklearn.exceptions import NotFittedError
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments


warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor_model"
PARAMS_FILE_NAME = "params.joblib"


class TextClassifier:
    """A wrapper class for the Gradient Boosting binary classifier.

    This class provides a consistent interface that can be used with other
    classifier models.
    """

    model_name = "DistilBERT Text Classifier"

    def __init__(
        self,
        num_classes: int,
        num_train_epochs: int = 3,
        train_batch_size: int = 8,
        **kwargs,
    ):
        """Construct a new DistilBERT text classifier.

        Args:

        """
        self.num_classes = num_classes
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size

        self.model = self.build_model()
        self._is_trained = False

    def build_model(self):
        """Build a new text classifier."""
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=self.num_classes,
        )

        return self.model

    def fit(self, train_dataset: Dataset) -> None:
        """Fit the text classifier to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        training_args = TrainingArguments(
            output_dir=paths.MODEL_ARTIFACTS_PATH,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.train_batch_size,  # batch size for training
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_steps=10,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        return self.model.predict(inputs)

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return self.model.predict_proba(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the binary classifier and return the accuracy.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The labels of the test data.
        Returns:
            float: The accuracy of the binary classifier.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the binary classifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        self.model.save_pretrained(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

        params = {
            "num_classes": self.num_classes,
            "num_train_epochs": self.num_train_epochs,
            "train_batch_size": self.train_batch_size,
        }
        joblib.dump(params, os.path.join(model_dir_path, PARAMS_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "TextClassifier":
        """Load the binary classifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Classifier: A new instance of the loaded binary classifier.
        """
        model = DistilBertForSequenceClassification.from_pretrained(
            os.path.join(model_dir_path, PREDICTOR_FILE_NAME)
        )
        params = joblib.load(os.path.join(model_dir_path, PARAMS_FILE_NAME))

        classifier = cls(**params)
        classifier.model = model
        return classifier

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name} ("


def train_predictor_model(
    train_dataset: Dataset, num_classes: int, hyperparameters: dict
) -> TextClassifier:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data labels.
        hyperparameters (dict): Hyperparameters for the classifier.

    Returns:
        'Classifier': The classifier model
    """
    classifier = TextClassifier(num_classes=num_classes, **hyperparameters)
    classifier.fit(train_dataset=train_dataset)
    return classifier


def predict_with_model(
    classifier: TextClassifier, data: pd.DataFrame, return_probs=False
) -> np.ndarray:
    """
    Predict class probabilities for the given data.

    Args:
        classifier (Classifier): The classifier model.
        data (pd.DataFrame): The input data.
        return_probs (bool): Whether to return class probabilities or labels.
            Defaults to True.

    Returns:
        np.ndarray: The predicted classes or class probabilities.
    """
    if return_probs:
        return classifier.predict_proba(data)
    return classifier.predict(data)


def save_predictor_model(model: TextClassifier, predictor_dir_path: str) -> None:
    """
    Save the classifier model to disk.

    Args:
        model (Classifier): The classifier model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> TextClassifier:
    """
    Load the classifier model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Classifier: A new instance of the loaded classifier model.
    """
    return TextClassifier.load(predictor_dir_path)


def evaluate_predictor_model(
    model: TextClassifier, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the classifier model and return the accuracy.

    Args:
        model (Classifier): The classifier model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the classifier model.
    """
    return model.evaluate(x_test, y_test)
