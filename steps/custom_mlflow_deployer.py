import os
import subprocess
import threading
import time
import mlflow
from typing import Any
from zenml import step
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

@step
def custom_mlflow_deployer(
    model: Any,
    deploy_decision: bool,
    model_name: str = "model",
    port: int = 5000,
) -> str:
    """Custom step to deploy MLflow model in a way that's compatible with Windows."""
    if not deploy_decision:
        print("Skipping deployment based on decision")
        return "not_deployed"
    
    # Find the latest MLflow run that contains our model
    mlflow_tracking_uri = get_tracking_uri()
    print(f"MLflow tracking URI: {mlflow_tracking_uri}")
    
    # Connect to MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Get the latest run with our model
    client = mlflow.tracking.MlflowClient()
    
    # Find the model in the model registry
    try:
        latest_model_version = client.get_latest_versions(model_name, stages=["None"])[0]
        run_id = latest_model_version.run_id
        print(f"Found model {model_name} with run_id: {run_id}")
    except Exception as e:
        print(f"Error finding model in registry: {e}")
        # Try finding the latest run directly
        runs = mlflow.search_runs()
        if len(runs) == 0:
            raise ValueError("No MLflow runs found")
        
        # Get the most recent run
        run_id = runs.iloc[0].run_id
        print(f"Using most recent run: {run_id}")
    
    # Construct the model URI
    model_uri = f"runs:/{run_id}/model"
    print(f"Model URI: {model_uri}")
    
    # Create a batch file that will run the MLflow server
    batch_file = os.path.join(os.getcwd(), "start_mlflow_server.bat")
    with open(batch_file, "w") as f:
        f.write(f'@echo off\n')
        f.write(f'echo Starting MLflow server for model: {model_name}\n')
        f.write(f'echo Model URI: {model_uri}\n')
        f.write(f'mlflow models serve -m "{model_uri}" --no-conda -p {port}\n')
        f.write(f'pause\n')  # So the window stays open if there's an error
    
    # Make the script executable
    os.chmod(batch_file, 0o755)
    
    print(f"Created MLflow server batch file at: {batch_file}")
    print(f"To start the server, run: {batch_file}")
    
    # Start the server in a new process window so it doesn't block
    # Note: shell=True is needed for Windows to open in a new window
    try:
        subprocess.Popen(f'start cmd /k "{batch_file}"', shell=True)
        print("MLflow server process started in a new window")
        # Give it a moment to start
        time.sleep(5)
    except Exception as e:
        print(f"Error starting MLflow server: {e}")
        return f"error: {str(e)}"
    
    # Return the endpoint information
    return f"http://127.0.0.1:{port}/invocations"