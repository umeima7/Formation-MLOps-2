import os
import time
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_model_with_io(features_path: str, model_registry_folder: str) -> None:
    features = pd.read_parquet(features_path)

    train_model(features, model_registry_folder)


def train_model(features: pd.DataFrame, model_registry_folder: str) -> None:
    target = 'Ba_avg'
    X = features.drop(columns=[target])
    y = features[target]
    model = RandomForestRegressor(n_estimators=1, max_depth=10, n_jobs=1)
    model.fit(X, y)
    joblib.dump(model, os.path.join(model_registry_folder, 'model.joblib'))
    # Création d'un nom unique avec timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    model_filename = f'model_{timestamp}.joblib'
    model_path = os.path.join(model_registry_folder, model_filename)

    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")
    return model_path
# Dossiers
MODEL_REGISTRY_FOLDER = "model_registry"
PREDICTIONS_FOLDER = "predictions"
os.makedirs(MODEL_REGISTRY_FOLDER, exist_ok=True)
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)

features_path = "/home/jovyan/work/Formation-MLOps-2/data/prepared_features_train.parquet"

model_path = train_model(pd.read_parquet(features_path), MODEL_REGISTRY_FOLDER)
print(f"Modèle entraîné et sauvegardé dans : {model_path}")


def predict_with_io(features_path: str, model_path: str, predictions_folder: str) -> None:
    features = pd.read_parquet(features_path)
    features = predict(features, model_path)
    time_str = time.strftime('%Y%m%d-%H%M%S')
    features['predictions_time'] = time_str
    features[['predictions', 'predictions_time']].to_csv(os.path.join(predictions_folder, time_str + '.csv'),
                                                         index=False)
    features[['predictions', 'predictions_time']].to_csv(os.path.join(predictions_folder, 'latest.csv'), index=False)


def predict(features: pd.DataFrame, model_path: str) -> pd.DataFrame:
    model = joblib.load(model_path)
    features['predictions'] = model.predict(features)
    return features 
