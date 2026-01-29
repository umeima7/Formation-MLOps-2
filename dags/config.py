import os

#PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_FOLDER = "/home/jovyan/work/Formation-MLOps-2"

DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
GENERATED_DATA_FOLDER = os.path.join(DATA_FOLDER, "generated_data")
PREDICTIONS_FOLDER = os.path.join(DATA_FOLDER, 'predictions')

TRAIN_DATA_PATH = os.path.join(DATA_FOLDER, 'la-haute-borne-data-2017-2020.csv')
# TODO change, should point to generator
GENERATED_DATA_PATH = os.path.join(DATA_FOLDER, 'la-haute-borne-data-2017-2020.csv')

FEATURES_PATH = os.path.join(DATA_FOLDER, 'prepared_features.parquet')

MODEL_REGISTRY_FOLDER = os.path.join(PROJECT_FOLDER, 'models')
MODEL_PATH = os.path.join(MODEL_REGISTRY_FOLDER, 'model.joblib')  # To change when needed
