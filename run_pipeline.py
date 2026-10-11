from pipelines.training_pipeline import train_pipeline
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from steps.config import ModelNameConfig
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":
    # 1️⃣ Ingest data
    raw_data = ingest_data()

    # 2️⃣ Clean data
    x_train, x_test, y_train, y_test = clean_data(data=raw_data)

    # 3️⃣ Define model config
    config = ModelNameConfig(
        model_name="lightgbm",  # Choose your model
        fine_tuning=False       # True if you want hyperparameter tuning
    )

    # 4️⃣ Train model
    model = train_model(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        config=config
    )

    # 5️⃣ Evaluate model
    evaluation(
        model=model,
        x_test=x_test,
        y_test=y_test
    )

    print(
        "Now run \n"
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline` experiment."
    )
