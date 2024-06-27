from collections import Counter
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ValidationError, field_validator


class ID(BaseModel):
    """
    A model representing the ID field of the dataset.
    """

    name: str
    description: str


class Target(BaseModel):
    """
    A model representing the target field of a text classification problem.
    """

    name: str
    description: str
    classes: List[str]

    @field_validator("classes")
    def target_classes_are_two_and_unique_and_not_empty_str(cls, target_classes):
        if len(target_classes) < 2:
            raise ValueError(
                f"Target classes must be a list with at least two labels."
                f"Given `{target_classes}` with length {len(target_classes)}"
            )
        if len(set(target_classes)) != len(target_classes):
            raise ValueError(
                "Target classes must be unique. " f"Given `{target_classes}`"
            )
        if "" in target_classes:
            raise ValueError(
                "Target classes must not contain empty strings. "
                f"Given `{target_classes}`"
            )
        return target_classes


class DataType(str, Enum):
    """Enum for the data type of a feature"""

    TEXT = "TEXT"
    CATEGORICAL = "CATEGORICAL"


class TextField(BaseModel):
    """
    A model representing the predictor fields in the dataset. Validates the
    presence and type of the 'example' field based on the 'dataType' field
    for NUMERIC dataType and presence and contents of the 'categories' field
    for CATEGORICAL dataType.
    """

    name: str
    description: str
    dataType: DataType
    example: Optional[str]


class SchemaModel(BaseModel):
    """
    A schema validator for multiclass classification problems. Validates the
    problem category, version, and predictor fields of the input schema.
    """

    title: str
    description: str = None
    modelCategory: str
    schemaVersion: float
    inputDataFormat: str = None
    encoding: str = None
    id: ID
    target: Target
    textField: TextField

    @field_validator("modelCategory")
    def valid_problem_category(cls, v):
        if v != "text_classification_base":
            raise ValueError(
                f"modelCategory must be 'text_classification_base'. Given {v}"
            )
        return v

    @field_validator("schemaVersion")
    def valid_version(cls, v):
        if v != 1.0:
            raise ValueError(f"schemaVersion must be set to 1.0. Given {v}")
        return v


def validate_schema_dict(schema_dict: dict) -> dict:
    """
    Validate the schema
    Args:
        schema_dict: dict
            data schema as a python dictionary

    Raises:
        ValueError: if the schema is invalid

    Returns:
        dict: validated schema as a python dictionary
    """
    try:
        schema_dict = SchemaModel.parse_obj(schema_dict).dict()
        return schema_dict
    except ValidationError as exc:
        raise ValueError(f"Invalid schema: {exc}") from exc
