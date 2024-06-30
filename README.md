# DistilBERT Text Classifier from HuggingFace (DistilBertForSequenceClassification)

DistilBERT Text Classifier for the text classification problem category as per Ready Tensor specifications.

## Project Description

This repository is a dockerized implementation of the re-usable text classifier model. It is implemented in flexible way so that it can be used with any text classification dataset with the use of CSV-formatted data, and a JSON-formatted data schema file. The main purpose of this repository is to provide a complete example of a machine learning model implementation that is ready for deployment.
The following are the requirements for using your data with this model:

- The data must be in CSV format.
- The schema file must contain an idField, textField and target columns.
- The train and test (or prediction) files must contain an ID field. The train data must also contain a target column.
- The train and test files can be either csv files or zipped csv files (.csv or .zip). 

---

Here are the highlights of this implementation: <br/>

- A **DistilBERT Text Classifier** algorithm built using **transformers** package.
  Additionally, the implementation contains the following features:
- **Data Validation**: Pydantic data validation is used for the schema, training and test files, as well as the inference request data.
- **Error handling and logging**: Python's logging module is used for logging and key functions include exception handling.

## Project Structure
The following is the directory structure of the project:
- **`examples/`**: This directory contains example files for the titanic dataset. Three files are included: `movie_reviews_schema.json`, `movie_reviews_train.csv` and `movie_reviews_test.csv`. You can place these files in the `inputs/schema`, `inputs/data/training` and `inputs/data/testing` folders, respectively.
- **`model_inputs_outputs/`**: This directory contains files that are either inputs to, or outputs from, the model. When running the model locally (i.e. without using docker), this directory is used for model inputs and outputs. This directory is further divided into:
  - **`/inputs/`**: This directory contains all the input files for this project, including the `data` and `schema` files. The `data` is further divided into `testing` and `training` subsets.
  - **`/model/artifacts/`**: This directory is used to store the model artifacts, such as trained models and their parameters.
  - **`/outputs/`**: The outputs directory contains sub-directories for error logs, and hyperparameter tuning outputs, and prediction results.
- **`src/`**: This directory holds the source code for the project. It is further divided into various subdirectories:
  - **`config/`**: for configuration files for data preprocessing, model hyperparameters, hyperparameter tuning-configuration specs, paths, etc.
  - **`data_models/`**: for data models for input validation including the schema, training and test files, and the inference request data. It also contains the data model for the batch prediction results.
  - **`schema/`**: for schema handler script. This script contains the class that provides helper getters/methods for the data schema.
  - **`prediction/`**: Scripts for the DistilBERT classifier model implemented using **transformers** python package.
  - **`serve.py`**: This script is used to serve the model as a REST API using **FastAPI**. It loads the artifacts and creates a FastAPI server to serve the model. It provides 2 endpoints: `/ping` and `/infer`. The `/ping` endpoint is used to check if the server is running. The `/infer` endpoint is used to make predictions.
  - **`serve_utils.py`**: This script contains utility functions used by the `serve.py` script.
  - **`logger.py`**: This script contains the logger configuration using **logging** module.
  - **`train.py`**: This script is used to train the model. It loads the data, preprocesses it, trains the model, and saves the artifacts in the path `./model_inputs_outputs/model/artifacts/`.
  - **`predict.py`**: This script is used to run batch predictions using the trained model. It loads the artifacts and creates and saves the predictions in a file called `predictions.csv` in the path `./model_inputs_outputs/outputs/predictions/`.
  - **`utils.py`**: This script contains utility functions used by the other scripts.
- **`.dockerignore`**: This file specifies the files and folders that should be ignored by Docker.
- **`.gitignore`**: This file specifies the files and folders that should be ignored by Git.
- **`docker-compose.yaml`**: This file is used to define the services that make up the application. It is used by Docker Compose to run the application.
- **`Dockerfile`**: This file is used to build the Docker image for the application.
- **`entry_point.sh`**: This file is used as the entry point for the Docker container. It is used to run the application. When the container is run using one of the commands `train`, `predict` or `serve`, this script runs the corresponding script in the `src` folder to execute the task.
- **`fix_line_endings.sh`**: This script is used to fix line endings in the project files. It is used to ensure that the project files have the correct line endings when the project is run on Windows.
- **`requirements.txt`**: This file contains the packages used in this project.
- **`LICENSE`**: This file contains the license for the project.
- **`README.md`**: This file (this particular document) contains the documentation for the project, explaining how to set it up and use it.

## Usage
In this section we cover the following:
- How to prepare your data for training and inference
- How to run the model implementation locally (without Docker)
- How to run the model implementation with Docker
- How to use the inference service (with or without Docker)
### Preparing your data
- If you plan to run this model implementation on your own dataset, you will need your training and testing data in a CSV format. Also, you will need to create a schema file as per the Ready Tensor specifications. The schema is in JSON format, and it's easy to create. You can use the example schema file provided in the `examples` directory as a template.
### To run locally (without Docker)
- Create your virtual environment and install dependencies listed in `requirements.txt`.
- Move the three example files (`movie_reviews_schema.json`, `movie_reviews_train.csv` and `movie_reviews_test.csv`) in the `examples` directory into the `./model_inputs_outputs/inputs/schema`, `./model_inputs_outputs/inputs/data/training` and `./model_inputs_outputs/inputs/data/testing` folders, respectively (or alternatively, place your custom dataset files in the same locations).
- Run the script `src/train.py` to train model. This will save the model artifacts, including the preprocessing pipeline and label encoder, in the path `./model_inputs_outputs/model/artifacts/`. If you want to run with hyperparameter tuning then include the `-t` flag. This will also save the hyperparameter tuning results in the path `./model_inputs_outputs/outputs/hpt_outputs/`.
- Run the script `src/predict.py` to run batch predictions using the trained model. This script will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `./model_inputs_outputs/outputs/predictions/`.
- Run the script `src/serve.py` to start the inference service, which can be queried using the `/ping` and `/infer` endpoints. The service runs on port 8080.
### To run with Docker
1. Set up a bind mount on host machine: It needs to mirror the structure of the `model_inputs_outputs` directory. Place the train data file in the `model_inputs_outputs/inputs/data/training` directory, the test data file in the `model_inputs_outputs/inputs/data/testing` directory, and the schema file in the `model_inputs_outputs/inputs/schema` directory.
2. Build the image. You can use the following command: <br/>
   `docker build -t classifier_img .` <br/>
   Here `classifier_img` is the name given to the container (you can choose any name).
3. Note the following before running the container for train, batch prediction or inference service:
   - The train, batch predictions tasks and inference service tasks require a bind mount to be mounted to the path `/opt/model_inputs_outputs/` inside the container. You can use the `-v` flag to specify the bind mount.
   - When you run the train or batch prediction tasks, the container will exit by itself after the task is complete. When the inference service task is run, the container will keep running until you stop or kill it.
   - When you run training task on the container, the container will save the trained model artifacts in the specified path in the bind mount. This persists the artifacts even after the container is stopped or killed.
   - When you run the batch prediction or inference service tasks, the container will load the trained model artifacts from the same location in the bind mount. If the artifacts are not present, the container will exit with an error.
   - The inference service runs on the container's port **8080**. Use the `-p` flag to map a port on local host to the port 8080 in the container.
   - Container runs as user 1000. Provide appropriate read-write permissions to user 1000 for the bind mount. Please follow the principle of least privilege when setting permissions. The following permissions are required:
     - Read access to the `inputs` directory in the bind mount. Write or execute access is not required.
     - Read-write access to the `outputs` directory and `model` directories. Execute access is not required.
4. Run training:
   - To run training, run the container with the following command container: <br/>
     `docker run -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs classifier_img train` <br/>
     where `classifier_img` is the name of the container. This will train the model and save the artifacts in the `model_inputs_outputs/model/artifacts` directory in the bind mount.
   
5. To run batch predictions, place the prediction data file in the `model_inputs_outputs/inputs/data/testing` directory in the bind mount. Then issue the command: <br/>
   `docker run -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs classifier_img predict` <br/>
   This will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `model_inputs_outputs/outputs/predictions/` in the bind mount.
6. To run the inference service, issue the following command on the running container: <br/>
   `docker run -p 8080:8080 -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs classifier_img serve` <br/>
   This starts the service on port 8080. You can query the service using the `/ping` and `/infer` endpoints. More information on the requests/responses on the endpoints is provided below.
### Using the Inference Service
#### Getting Predictions
To get predictions for a single sample, use the following command:
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  {
    "instances": [
        {
            "id": "0",
            "text": "some text"
        },
        {
            "id": "1",
            "text": "this is a very positive text"
        },
        {
            "id": "2",
            "text": "this is a very negative text"
        }

    ]
}' http://localhost:8080/infer
```
The key `instances` contains a list of objects, each of which is a sample for which the prediction is requested. The server will respond with a JSON object containing the predicted probabilities for each input record:
```json
{
    "status": "success",
    "message": "",
    "timestamp": "2024-06-30T15:14:41.320863",
    "requestId": "0bf266d354",
    "targetClasses": [
        "neg",
        "pos"
    ],
    "targetDescription": "..",
    "predictions": [
        {
            "sampleId": "0",
            "predictedClass": "pos",
            "predictedProbabilities": [
                0.47454,
                0.52546
            ]
        },
        {
            "sampleId": "1",
            "predictedClass": "pos",
            "predictedProbabilities": [
                0.07711,
                0.92289
            ]
        },
        {
            "sampleId": "2",
            "predictedClass": "neg",
            "predictedProbabilities": [
                0.67533,
                0.32467
            ]
        }
    ]
}
```

## Requirements

Dependencies for the main model implementation in `src` are listed in the file `requirements.txt`.
You can install these packages by running the following command from the root of your project directory:

```python
pip install -r requirements.txt
```

## LICENSE

This project is provided under the Apache 2.0 License. Please see the [LICENSE](LICENSE) file for more information.

## Contact Information

Repository created by Ready Tensor, Inc. (https://www.readytensor.ai/)