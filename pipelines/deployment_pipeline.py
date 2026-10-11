import json
import os
import numpy as np
import pandas as pd
from pydantic import BaseModel

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from steps.config import ModelNameConfig
from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])
requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")


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


@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline."""
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # Prefer running services
    services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=True,
    )

    # Fallback to non-running services
    if not services:
        services = model_deployer.find_model_server(
            pipeline_name=pipeline_name,
            pipeline_step_name=pipeline_step_name,
            model_name=model_name,
            running=False,
        )

    if not services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by '{pipeline_step_name}' "
            f"in pipeline '{pipeline_name}' for model name '{model_name}'.\n"
            f"Run the deployment first, e.g.: python run_deployment.py --config deploy --force-deploy"
        )

    return services[0]


@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run inference against a prediction service."""
    # Ensure service is running; this will start it if it was stopped.
    service.start(timeout=10)

    data = json.loads(data)
    data.pop("columns", None)
    data.pop("index", None)

    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]

    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array = np.array(json_list)
    prediction = service.predict(data_array)
    return prediction


# ------------------ Pipelines ------------------

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    min_accuracy: float = 2.0,  # raised so RMSE ~1.32 passes and deploys
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
    model_config: ModelNameConfig = ModelNameConfig(model_name="lightgbm", fine_tuning=False),
    force_deploy: bool = False,
):
    """Full deployment pipeline linking all steps."""
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

    # 5) Deploy if decision is True - no daemon parameter
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=final_decision,
        workers=workers,
        timeout=timeout,
        model_name="model",
    )


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    """Inference pipeline using the deployed model."""
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name="model",
    )
    predictor(service=model_deployment_service, data=batch_data)