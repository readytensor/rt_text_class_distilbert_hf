from config import paths

from data_models.data_validator import validate_data
from logger import get_logger, log_error

from prediction.predictor_model import (
    save_predictor_model,
    train_predictor_model,
)

from schema.data_schema import load_json_data_schema, save_schema
from utils import (
    read_csv_in_directory,
    read_json_as_dict,
    set_seeds,
    save_json,
    load_hf_dataset,
    label_encoding,
)

logger = get_logger(task_name="train")

import sys


def run_training(
    input_schema_dir: str = paths.INPUT_SCHEMA_DIR,
    saved_schema_dir_path: str = paths.SAVED_SCHEMA_DIR_PATH,
    model_config_file_path: str = paths.MODEL_CONFIG_FILE_PATH,
    train_dir: str = paths.TRAIN_DIR,
    label_encoding_map_file_path: str = paths.LABEL_ENCODING_MAP_FILE_PATH,
    predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
    default_hyperparameters_file_path: str = paths.DEFAULT_HYPERPARAMETERS_FILE_PATH,
    saved_tokenizer_dir_path: str = paths.SAVED_TOKENIZER_DIR_PATH,
) -> None:
    """
    Run the training process and saves model artifacts

    Args:
        input_schema_dir (str, optional): The directory path of the input schema.
        saved_schema_dir_path (str, optional): The path where to save the schema.
        model_config_file_path (str, optional): The path of the model
            configuration file.
        train_dir (str, optional): The directory path of the train data.
        label_encoding_map_file_path (str, optional): The path of the label encoding file.
        predictor_dir_path (str, optional): Dir path where to save the
            predictor model.
        default_hyperparameters_file_path (str, optional): The path of the default
            hyperparameters file.
    Returns:
        None
    """

    try:

        logger.info("Starting training...")
        # load and save schema
        logger.info("Loading and saving schema...")
        data_schema = load_json_data_schema(input_schema_dir)
        save_schema(schema=data_schema, save_dir_path=saved_schema_dir_path)

        # load model config
        logger.info("Loading model config...")
        model_config = read_json_as_dict(model_config_file_path)

        # set seeds
        logger.info("Setting seeds...")
        set_seeds(seed_value=model_config["seed_value"])

        # load train data
        logger.info("Loading train data...")
        train_data = read_csv_in_directory(file_dir_path=train_dir)

        # validate the data
        logger.info("Validating train data...")
        train_data = validate_data(
            data=train_data, data_schema=data_schema, is_train=True
        )

        # target encoding
        train_data, label_encoding_map = label_encoding(train_data, data_schema.target)
        save_json(label_encoding_map_file_path, label_encoding_map)

        train_data, tokenizer = load_hf_dataset(
            train_data, data_schema.text_field, data_schema.target
        )

        logger.info("Saving tokenizer...")
        tokenizer.save_pretrained(saved_tokenizer_dir_path)

        logger.info("Training classifier...")
        default_hyperparameters = read_json_as_dict(default_hyperparameters_file_path)
        predictor = train_predictor_model(
            train_data,
            num_classes=len(data_schema.target_classes),
            hyperparameters=default_hyperparameters,
        )

        # save predictor model
        logger.info("Saving classifier...")
        save_predictor_model(predictor, predictor_dir_path)

        logger.info("Training completed successfully")

    except Exception as exc:
        err_msg = "Error occurred during training."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.TRAIN_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_training()
