import os
import time
import random
from dataclasses import dataclass
from typing import Optional
from temporalio import activity

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


@dataclass
class DatasetInfo:
    num_samples: int
    num_features: int
    num_classes: int
    train_size: int
    test_size: int


@dataclass
class TrainingConfig:
    n_estimators: int = 100
    max_depth: Optional[int] = None
    random_state: int = 42
    test_size: float = 0.2


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    model_path: str


@activity.defn
async def load_and_validate_data(dataset_name: str) -> DatasetInfo:
    """
    Activity 1: Load and validate a dataset.
    In a real pipeline, this would fetch from S3, a database, or a data lake.
    """
    activity.logger.info(f"Loading dataset: {dataset_name}")
    activity.heartbeat(f"Loading dataset: {dataset_name}")

    # Simulate dataset loading (using sklearn's make_classification for demo)
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=42,
    )

    activity.heartbeat("Validating data quality...")
    time.sleep(1)  # Simulate I/O time

    # Save to a temp file so downstream activities can use it
    data_dir = "/tmp/temporal_ml_pipeline"
    os.makedirs(data_dir, exist_ok=True)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    df.to_csv(f"{data_dir}/{dataset_name}.csv", index=False)

    info = DatasetInfo(
        num_samples=len(X),
        num_features=X.shape[1],
        num_classes=len(np.unique(y)),
        train_size=int(len(X) * 0.8),
        test_size=int(len(X) * 0.2),
    )
    activity.logger.info(f"Dataset loaded: {info}")
    return info


@activity.defn
async def preprocess_data(dataset_name: str, config: TrainingConfig) -> dict:
    """
    Activity 2: Preprocess the data.
    Split into train/test, apply feature scaling.
    """
    activity.logger.info("Preprocessing data...")
    activity.heartbeat("Splitting dataset...")

    data_dir = "/tmp/temporal_ml_pipeline"
    df = pd.read_csv(f"{data_dir}/{dataset_name}.csv")

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    activity.heartbeat("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    time.sleep(0.5)

    np.save(f"{data_dir}/X_train.npy", X_train_scaled)
    np.save(f"{data_dir}/X_test.npy", X_test_scaled)
    np.save(f"{data_dir}/y_train.npy", y_train)
    np.save(f"{data_dir}/y_test.npy", y_test)

    activity.logger.info(f"Preprocessing complete. Train: {len(X_train)}, Test: {len(X_test)}")
    return {"train_size": len(X_train), "test_size": len(X_test)}


@activity.defn
async def train_model(config: TrainingConfig) -> str:
    """
    Activity 3: Train the ML model.
    Uses heartbeats so Temporal knows it's still alive during long training.
    """
    activity.logger.info("Starting model training...")
    data_dir = "/tmp/temporal_ml_pipeline"
    model_dir = "/tmp/temporal_ml_pipeline/models"
    os.makedirs(model_dir, exist_ok=True)

    X_train = np.load(f"{data_dir}/X_train.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")

    activity.heartbeat("Initializing RandomForest model...")

    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state,
    )

    # Simulate training in chunks with heartbeats
    chunk_size = config.n_estimators // 5
    for i in range(5):
        partial_estimators = (i + 1) * chunk_size
        activity.heartbeat(f"Training progress: {partial_estimators}/{config.n_estimators} trees")
        time.sleep(0.3)

    model.fit(X_train, y_train)

    model_path = f"{model_dir}/rf_model.pkl"
    import pickle
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    activity.logger.info(f"Model trained and saved to: {model_path}")
    return model_path


@activity.defn
async def evaluate_model(model_path: str) -> ModelMetrics:
    """
    Activity 4: Evaluate the trained model on the test set.
    """
    activity.logger.info("Evaluating model...")
    activity.heartbeat("Loading model and test data...")

    data_dir = "/tmp/temporal_ml_pipeline"
    X_test = np.load(f"{data_dir}/X_test.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")

    import pickle
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    activity.heartbeat("Running predictions...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = ModelMetrics(
        accuracy=round(accuracy, 4),
        precision=round(report["weighted avg"]["precision"], 4),
        recall=round(report["weighted avg"]["recall"], 4),
        f1_score=round(report["weighted avg"]["f1-score"], 4),
        model_path=model_path,
    )
    activity.logger.info(f"Evaluation complete: {metrics}")
    return metrics


@activity.defn
async def register_model(metrics: ModelMetrics, model_name: str) -> str:
    """
    Activity 5: Register the model in a model registry.
    In a real pipeline, this would push to MLflow, SageMaker, or Vertex AI.
    """
    activity.logger.info(f"Registering model: {model_name}")
    activity.heartbeat("Checking model quality gate...")

    time.sleep(0.5)

    # Quality gate: only register if accuracy > 0.85
    if metrics.accuracy < 0.85:
        from temporalio.exceptions import ApplicationError
        raise ApplicationError(
            f"Model failed quality gate: accuracy={metrics.accuracy} < 0.85",
            non_retryable=True,
        )

    registry_entry = {
        "model_name": model_name,
        "version": "1.0",
        "accuracy": metrics.accuracy,
        "f1_score": metrics.f1_score,
        "artifact_path": metrics.model_path,
        "status": "registered",
    }

    activity.logger.info(f"Model registered successfully: {registry_entry}")
    return f"Model '{model_name}' v1.0 registered with accuracy={metrics.accuracy}"
