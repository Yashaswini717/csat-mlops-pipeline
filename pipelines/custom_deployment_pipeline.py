import json
import os
import numpy as np
import pandas as pd
from pydantic import BaseModel

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW

from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from steps.config import ModelNameConfig
from steps.custom_mlflow_deployer import custom_mlflow_deployer
from steps.custom_mlflow_predictor import custom_mlflow_predictor
from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])

# ------------------ Steps ------------------

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    return data


class DeploymentTriggerConfig(BaseModel):
    # Threshold value for the metric used to decide deployment
    threshold: float = 0.9
    # If True, deploy when metric < threshold (e.g., RMSE/MSE).
    # If False, deploy when metric > threshold (e.g., accuracy/R2).
    lower_is_better: bool = True


@step
def deployment_trigger(
    metric: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """Decides if the metric is good enough to deploy."""
    return metric < config.threshold if config.lower_is_better else metric > config.threshold


@step
def combine_decision(deploy_decision: bool, force_deploy: bool) -> bool:
    """Combine threshold decision with a manual override flag."""
    decision = deploy_decision or force_deploy
    print(f"[deploy] Deploy decision: {decision} (metric:{deploy_decision}, force:{force_deploy})")
    return decision


# ------------------ Pipelines ------------------

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def custom_deployment_pipeline(
    min_accuracy: float = 2.0,
    port: int = 5000,
    model_config: ModelNameConfig = ModelNameConfig(model_name="lightgbm", fine_tuning=False),
    force_deploy: bool = False,
):
    """Custom deployment pipeline using Windows-compatible approach."""
    # 1) Ingest and clean data
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)

    # 2) Train model (pass model_config)
    model = train_model(x_train, x_test, y_train, y_test, config=model_config)

    # 3) Evaluate (returns mse, rmse)
    mse, rmse = evaluation(model, x_test, y_test)
    print(f"[deploy] RMSE: {rmse}, Threshold: {min_accuracy}")

    # 4) Decide deployment: for regression, lower RMSE is better.
    decision_from_metric = deployment_trigger(
        metric=rmse,
        config=DeploymentTriggerConfig(
            threshold=min_accuracy,    # deploy if rmse < min_accuracy
            lower_is_better=True
        ),
    )

    final_decision = combine_decision(decision_from_metric, force_deploy)

    # 5) Use custom deployer instead of ZenML's MLflow deployer
    endpoint = custom_mlflow_deployer(
        model=model,
        deploy_decision=final_decision,
        model_name="model",
        port=port,
    )
    
    return endpoint


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def custom_inference_pipeline(model_endpoint: str):
    """Custom inference pipeline using the manually deployed model."""
    batch_data = dynamic_importer()
    predictions = custom_mlflow_predictor(data=batch_data, model_endpoint=model_endpoint)
    return predictions