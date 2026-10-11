import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from lightgbm import LGBMRegressor
import mlflow
import mlflow.lightgbm

from steps.config import ModelNameConfig

# Use your experiment tracker name from the stack: "mlflow_tracker"
@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig = ModelNameConfig(),
) -> RegressorMixin:
    """Train a LightGBM model, log it to MLflow as 'model', and return the trained model."""
    try:
        if config.model_name.lower() != "lightgbm":
            logging.warning(
                f"Model name '{config.model_name}' is not supported. "
                "Using LightGBM by default."
            )

        model = LGBMRegressor()
        model.fit(x_train, y_train)

        # Explicitly log to MLflow so the deployer step can find it in THIS run.
        # Keep the name 'model' to match the deployer and the loader.
        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path="model",
            registered_model_name="model",
        )

        logging.info("LightGBM model training and MLflow logging completed successfully.")
        return model

    except Exception as e:
        logging.error(f"Error in train_model step: {e}")
        raise e