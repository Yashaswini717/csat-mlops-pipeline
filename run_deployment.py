import click
from rich import print
from typing import cast

from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline
from steps.config import ModelNameConfig
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService

# CLI options
DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Run deployment only (`deploy`), inference only (`predict`), or both (`deploy_and_predict`).",
)
@click.option(
    "--min-accuracy",
    default=2.0,
    help="Maximum RMSE threshold to deploy (lower is better). Example: 2.0 means deploy if RMSE < 2.0.",
)
@click.option("--force-deploy", is_flag=True, help="Ignore metric threshold and force deployment.")
@click.option("--model-name", default="lightgbm", help="Model to train: lightgbm, xgboost, randomforest, linear_regression")
@click.option("--fine-tuning", is_flag=True, help="Enable fine-tuning for the model")
def main(config: str, min_accuracy: float, force_deploy: bool, model_name: str, fine_tuning: bool):
    """Run the MLflow deployment and inference pipelines."""

    # Create model configuration
    model_config = ModelNameConfig(model_name=model_name, fine_tuning=fine_tuning)

    # Get the MLFlow model deployer component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()

    deploy = config in [DEPLOY, DEPLOY_AND_PREDICT]
    predict = config in [PREDICT, DEPLOY_AND_PREDICT]

    if deploy:
        print(f"[bold green]Running deployment pipeline with {model_name}, fine_tuning={fine_tuning}[/bold green]")
        continuous_deployment_pipeline(
            min_accuracy=min_accuracy,
            workers=3,
            timeout=60,
            model_config=model_config,
            force_deploy=force_deploy,
        )

    if predict:
        print("[bold cyan]Running inference pipeline[/bold cyan]")
        inference_pipeline(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
        )

    # MLflow UI instructions
    print(
        "You can run:\n"
        f"[italic green]    mlflow ui --backend-store-uri '{get_tracking_uri()}'[/italic green]\n"
        "...to inspect your experiment runs within the MLflow UI.\n"
    )

    # Check deployed services
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(f"The MLflow prediction server is running at: {service.prediction_url}")
        elif service.is_failed:
            print(
                f"The MLflow prediction server failed:\n"
                f"Last state: '{service.status.state.value}'\n"
                f"Last error: '{service.status.last_error}'"
            )
        else:
            print("A deployed MLflow service exists but is not running yet. It will be started by the inference pipeline.")
    else:
        print("No MLflow prediction server is found. Run with `--config deploy` first, or ensure your deployment trigger condition is met.")


if __name__ == "__main__":
    main()