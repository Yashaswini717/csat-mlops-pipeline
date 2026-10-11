import click
from rich import print
from typing import cast

from pipelines.custom_deployment_pipeline import custom_deployment_pipeline, custom_inference_pipeline
from steps.config import ModelNameConfig

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
@click.option("--port", default=5000, help="Port for the MLflow server")
@click.option("--model-name", default="lightgbm", help="Model to train: lightgbm, xgboost, randomforest, linear_regression")
@click.option("--fine-tuning", is_flag=True, help="Enable fine-tuning for the model")
@click.option("--endpoint", default=None, help="Prediction endpoint URL (for predict only)")
def main(config: str, min_accuracy: float, force_deploy: bool, port: int, model_name: str, fine_tuning: bool, endpoint: str):
    """Run the custom MLflow deployment and inference pipelines."""

    # Create model configuration
    model_config = ModelNameConfig(model_name=model_name, fine_tuning=fine_tuning)

    deploy = config in [DEPLOY, DEPLOY_AND_PREDICT]
    predict = config in [PREDICT, DEPLOY_AND_PREDICT]
    
    model_endpoint = endpoint
    
    if deploy:
        print(f"[bold green]Running custom deployment pipeline with {model_name}, fine_tuning={fine_tuning}[/bold green]")
        pipeline_response = custom_deployment_pipeline(
            min_accuracy=min_accuracy,
            port=port,
            model_config=model_config,
            force_deploy=force_deploy,
        )
        
        # Use a fixed endpoint since we know where MLflow will be running
        model_endpoint = f"http://127.0.0.1:{port}/invocations"
        print(f"[bold green]Model deployed at endpoint: {model_endpoint}[/bold green]")

    if predict:
        if not model_endpoint:
            print("[bold red]Error: No endpoint specified for prediction[/bold red]")
            print("Use --endpoint or deploy the model first")
            return
            
        print(f"[bold cyan]Running custom inference pipeline against endpoint: {model_endpoint}[/bold cyan]")
        prediction_response = custom_inference_pipeline(model_endpoint=model_endpoint)
        
        # Print a message about checking the prediction results
        print("[bold green]Prediction completed! Check ZenML dashboard for results.[/bold green]")
        print("[bold yellow]You can also query predictions directly from the MLflow server at:[/bold yellow]")
        print(f"[bold yellow]POST {model_endpoint} with JSON payload[/bold yellow]")


if __name__ == "__main__":
    main()
    