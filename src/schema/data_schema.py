import os
from typing import Dict, List, Tuple

import joblib

from data_models.schema_validator import validate_schema_dict
from utils import read_json_as_dict

SCHEMA_FILE_NAME = "schema.joblib"


class TextClassificationSchema:
    """
    A class for loading and providing access to a multiclass classification schema.

    This class allows users to work with a generic schema for multiclass classification
    problems, enabling them to create algorithm implementations that are not hardcoded
    to specific feature names. The class provides methods to retrieve information about
    the schema, such as the ID field, target field, allowed values for the target
    field, and details of the features (categorical and numeric). This makes it easier
    to preprocess and manipulate the input data according to the schema, regardless of
    the specific dataset used.
    """

    def __init__(self, schema_dict: dict) -> None:
        """
        Initializes a new instance of the `TextClassificationSchema` class
        and using the schema dictionary.

        Args:
            schema_dict (dict): The python dictionary of schema.
        """
        self.schema = schema_dict

    @property
    def model_category(self) -> str:
        """
        Gets the model category.

        Returns:
            str: The category of the machine learning model
                (e.g., multiclass_classification, multi-class_classification,
                regression, object_detection, etc.).
        """
        return self.schema["modelCategory"]

    @property
    def title(self) -> str:
        """
        Gets the title of the dataset or problem.

        Returns:
            str: The title of the dataset or the problem.
        """
        return self.schema["title"]

    @property
    def description(self) -> str:
        """
        Gets the description of the dataset or problem.

        Returns:
            str: A brief description of the dataset or the problem.
        """
        return self.schema["description"]

    @property
    def schema_version(self) -> float:
        """
        Gets the version number of the schema.

        Returns:
            float: The version number of the schema.
        """
        return self.schema["schemaVersion"]

    @property
    def input_data_format(self) -> str:
        """
        Gets the format of the input data.

        Returns:
            str: The format of the input data (e.g., CSV, JSON, etc.).
        """
        return self.schema["inputDataFormat"]

    @property
    def encoding(self) -> str:
        """
        Gets the encoding of the input data.

        Returns:
            str: The encoding of the input data (e.g., "utf-8", "iso-8859-1", etc.).
        """
        return self.schema["encoding"]

    @property
    def id(self) -> str:
        """
        Gets the name of the ID field.

        Returns:
            str: The name of the ID field.
        """
        return self.schema["id"]["name"]

    @property
    def id_description(self) -> str:
        """
        Gets the description for the ID field.

        Returns:
            str: The description for the ID field.
        """
        return self.schema["id"].get(
            "description", "No description for target available."
        )

    @property
    def target(self) -> str:
        """
        Gets the name of the target field.

        Returns:
            str: The name of the target field.
        """
        return self.schema["target"]["name"]

    @property
    def target_classes(self) -> List[str]:
        """
        Gets the classes for the target field.

        Returns:
            List[str]: The list of allowed classes for the target field.
        """
        return [str(c) for c in self.schema["target"]["classes"]]

    @property
    def text_field(self) -> str:
        """
        Gets the name of the text field.

        Returns:
            str: The name of the text field.
        """
        return self.schema["textField"]["name"]

    @property
    def target_description(self) -> str:
        """
        Gets the description for the target field.

        Returns:
            str: The description for the target field.
        """
        return self.schema["target"].get(
            "description", "No description for target available."
        )


def load_json_data_schema(schema_dir_path: str) -> TextClassificationSchema:
    """
    Load the JSON file schema into a dictionary, validate the schema dict for
    its correctness, and use the validated schema to instantiate the schema provider.

    Args:
    - schema_dir_path (str): Path from where to read the schema json file.

    Returns:
        TextClassificationSchema: An instance of the
                                        TextClassificationSchema.
    """
    schema_dict = read_json_as_dict(input_path=schema_dir_path)
    validated_schema_dict = validate_schema_dict(schema_dict=schema_dict)
    data_schema = TextClassificationSchema(validated_schema_dict)
    return data_schema


def save_schema(schema: TextClassificationSchema, save_dir_path: str) -> None:
    """
    Save the schema to a JSON file.

    Args:
        schema (TextClassificationSchema): The schema to be saved.
        save_dir_path (str): The dir path to save the schema to.
    """
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    file_path = os.path.join(save_dir_path, SCHEMA_FILE_NAME)
    joblib.dump(schema, file_path)


def load_saved_schema(save_dir_path: str) -> TextClassificationSchema:
    """
    Load the saved schema from a JSON file.

    Args:
        save_dir_path (str): The path to load the schema from.

    Returns:
        TextClassificationSchema: An instance of the
                                        TextClassificationSchema.
    """
    file_path = os.path.join(save_dir_path, SCHEMA_FILE_NAME)
    if not os.path.exists(file_path):
        print("no such file")
        raise FileNotFoundError(f"No such file or directory: '{file_path}'")
    return joblib.load(file_path)
