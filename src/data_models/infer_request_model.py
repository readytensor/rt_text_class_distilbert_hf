from typing import List

from pydantic import BaseModel, Field, create_model

from schema.data_schema import TextClassificationSchema


def create_instance_model(schema: TextClassificationSchema) -> BaseModel:
    """
    Creates a dynamic Pydantic model for instance validation based on the schema.

    Args:
        schema (TextClassificationSchema): The multiclass classification schema.

    Returns:
        BaseModel: The dynamically created Pydantic model.
    """
    fields = {schema.id: (str, Field(..., example="some_id_123"))}

    text_field = schema.text_field
    fields[text_field] = (str, Field(..., example="some text"))

    return create_model("Instance", **fields)


def get_inference_request_body_model(
    schema: TextClassificationSchema,
) -> BaseModel:
    """
    Creates a dynamic Pydantic model for the inference request body validation based
    on the schema.

    It ensures that the request body contains a list of instances, each of which is a
    dictionary representing a data instance with the text field specified in the schema.

    Args:
        schema (TextClassificationSchema): The multiclass classification schema.

    Returns:
        BaseModel: The dynamically created Pydantic model.
    """
    InstanceModel = create_instance_model(schema)

    class InferenceRequestBody(BaseModel):
        """
        InferenceRequestBody is a Pydantic model for validating the request body of an
            inference endpoint.

        The following validations are performed on the request data:
            - The request body contains a key 'instances' with a list of dictionaries
                as its value.
            - The list is not empty (i.e., at least one instance must be provided).
            - Each instance contains the ID field whose name is defined in the
                schema file.
            - Each instance contains the text field whose name is defined in the schema file.

        Attributes:
            instances (List[Instance_Model]): A list of data instances to be validated.
        """

        instances: List[InstanceModel] = Field(..., min_items=1)

    return InferenceRequestBody
