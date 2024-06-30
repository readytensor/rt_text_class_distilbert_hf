from typing import List

import numpy as np
import pandas as pd

from config import paths
from data_models.data_validator import validate_data
from data_models.prediction_data_model import validate_predictions
from logger import get_logger, log_error
from prediction.predictor_model import load_predictor_model, predict_with_model
from schema.data_schema import load_saved_schema
from utils import (
    read_csv_in_directory,
    read_json_as_dict,
    save_dataframe_as_csv,
    load_hf_dataset,
    get_sorted_class_names,
)

logger = get_logger(task_name="predict")


def create_predictions_dataframe(
    predictions_arr: np.ndarray,
    class_names: List[str],
    prediction_field_name: str,
    ids: pd.Series,
    id_field_name: str,
    return_probs: bool = False,
) -> pd.DataFrame:
    """
    Converts the predictions numpy array into a dataframe having the required structure.

    Performs the following transformations:
    - converts to pandas dataframe
    - adds class labels as headers for columns containing predicted probabilities
    - inserts the id column

    Args:
        predictions_arr (np.ndarray): Predicted probabilities from predictor model.
        class_names List[str]: List of target classes (labels).
        prediction_field_name (str): Field name to use for predicted class.
        ids: ids as a numpy array for each of the samples in  predictions.
        id_field_name (str): Name to use for the id field.
        return_probs (bool, optional): If True, returns the predicted probabilities
            for each class. If False, returns the final predicted class for each
            data point. Defaults to False.

    Returns:
        Predictions as a pandas dataframe
    """
    if predictions_arr.shape[1] != len(class_names):
        raise ValueError(
            "Length of class names does not match number of prediction columns"
        )
    predictions_df = pd.DataFrame(predictions_arr, columns=class_names)
    if len(predictions_arr) != len(ids):
        raise ValueError("Length of ids does not match number of predictions")
    predictions_df.insert(0, id_field_name, ids)
    if return_probs:
        return predictions_df
    predictions_df[prediction_field_name] = predictions_df[class_names].idxmax(axis=1)
    predictions_df.drop(class_names, axis=1, inplace=True)
    return predictions_df


def run_batch_predictions(
    saved_schema_dir_path: str = paths.SAVED_SCHEMA_DIR_PATH,
    model_config_file_path: str = paths.MODEL_CONFIG_FILE_PATH,
    test_dir: str = paths.TEST_DIR,
    predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
    predictions_file_path: str = paths.PREDICTIONS_FILE_PATH,
    label_encoding_map_file_path: str = paths.LABEL_ENCODING_MAP_FILE_PATH,
    saved_tokenizer_dir_path: str = paths.SAVED_TOKENIZER_DIR_PATH,
) -> None:
    """
    Run batch predictions on test data, save the predicted probabilities to a CSV file.

    This function reads test data from the specified directory,
    loads the preprocessing pipeline and pre-trained predictor model,
    transforms the test data using the pipeline,
    makes predictions using the trained predictor model,
    adds ids into the predictions dataframe,
    and saves the predictions as a CSV file.

    Args:
        saved_schema_dir_path (str): Dir path to the saved data schema.
        model_config_file_path (str): Path to the model configuration file.
        test_dir (str): Directory path for the test data.
        predictor_file_path (str): Path to the saved predictor model file.
        predictions_file_path (str): Path where the predictions file will be saved.
        label_encoding_map_file_path (str): Path to the label encoding map file.
        saved_tokenizer_dir_path (str): Path to the saved tokenizer directory.
    """

    try:
        logger.info("Making batch predictions...")

        logger.info("Loading schema...")
        data_schema = load_saved_schema(saved_schema_dir_path)

        logger.info("Loading model config...")
        model_config = read_json_as_dict(model_config_file_path)

        logger.info("Loading prediction input data...")
        test_data = read_csv_in_directory(file_dir_path=test_dir)

        # validate the data
        logger.info("Validating prediction data...")
        test_data = validate_data(
            data=test_data, data_schema=data_schema, is_train=False
        )

        test_data, _ = load_hf_dataset(
            test_data,
            data_schema.text_field,
            data_schema.target,
            is_train=False,
            tokenizer_dir_path=saved_tokenizer_dir_path,
        )

        logger.info("Loading predictor model...")
        predictor_model = load_predictor_model(predictor_dir_path)

        logger.info("Making predictions...")
        predictions_arr = predict_with_model(
            predictor_model, test_data, return_probs=True
        )

        class_names = get_sorted_class_names(label_encoding_map_file_path)

        logger.info("Transforming predictions into dataframe...")
        predictions_df = create_predictions_dataframe(
            predictions_arr,
            class_names,
            model_config["prediction_field_name"],
            test_data[data_schema.id],
            data_schema.id,
            return_probs=True,
        )

        logger.info("Validating predictions...")
        validated_predictions = validate_predictions(predictions_df, data_schema)

        logger.info("Saving predictions...")
        save_dataframe_as_csv(
            dataframe=validated_predictions, file_path=predictions_file_path
        )

        logger.info("Batch predictions completed successfully")

    except Exception as exc:
        err_msg = "Error occurred during prediction."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.PREDICT_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_batch_predictions()
