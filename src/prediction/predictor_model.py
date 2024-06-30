import os
import warnings
import torch
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import paths
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.exceptions import NotFittedError
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments


warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor_model"
PARAMS_FILE_NAME = "params.joblib"

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class TextClassifier:
    """A wrapper class for the text classifier.

    This class provides a consistent interface that can be used with other
    classifier models.
    """

    model_name = "DistilBERT Text Classifier"

    def __init__(
        self,
        num_classes: int,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        **kwargs,
    ):
        """Construct a new DistilBERT text classifier.

        Args:
            num_classes (int): The number of classes in the dataset.
            num_train_epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            learning_rate (float): The learning rate for training.
            **kwargs: Additional keyword arguments.

        """
        self.num_classes = num_classes
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kwargs = kwargs

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
            train_inputs (Dataset): The features of the training data.
        """
        training_args = TrainingArguments(
            output_dir=paths.MODEL_ARTIFACTS_PATH,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            logging_steps=10,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()
        self._is_trained = True

    def predict(self, input_dataset: Dataset, return_probs: bool = True) -> np.ndarray:
        """Predict class labels for the given data.

        Args:
            inputs (Dataset): The input data.
            return_probs (bool): Whether to return class probabilities or labels.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        self.model.eval()
        self.model.to(device)
        dataloader = DataLoader(input_dataset, batch_size=self.batch_size)

        all_probs = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids = torch.stack(batch["input_ids"], dim=1).to(self.model.device)
                attention_mask = torch.stack(batch["attention_mask"], dim=1).to(
                    self.model.device
                )
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                all_probs.append(probabilities.cpu().tolist())

        if return_probs:
            return np.concatenate(all_probs, axis=0)

        return np.argmax(probabilities, axis=1)

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
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
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
        train_X (Dataset): The training data inputs.
        num_classes (int): The number of classes in the dataset.
        hyperparameters (dict): Hyperparameters for the classifier.

    Returns:
        'TextClassifier': The classifier model
    """
    classifier = TextClassifier(num_classes=num_classes, **hyperparameters)
    classifier.fit(train_dataset=train_dataset)
    return classifier


def predict_with_model(
    classifier: TextClassifier, data: Dataset, return_probs=True
) -> np.ndarray:
    """
    Predict class probabilities for the given data.

    Args:
        classifier (TextClassifier): The classifier model.
        data (Dataset): The input data.
        return_probs (bool): Whether to return class probabilities or labels.
            Defaults to True.

    Returns:
        np.ndarray: The predicted classes or class probabilities.
    """
    return classifier.predict(data, return_probs=return_probs)


def save_predictor_model(model: TextClassifier, predictor_dir_path: str) -> None:
    """
    Save the classifier model to disk.

    Args:
        model (TextClassifier): The classifier model to save.
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
        TextClassifier: A new instance of the loaded classifier model.
    """
    return TextClassifier.load(predictor_dir_path)


def evaluate_predictor_model(
    model: TextClassifier, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the classifier model and return the accuracy.

    Args:
        model (TextClassifier): The classifier model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the classifier model.
    """
    return model.evaluate(x_test, y_test)
