import json
import time
import requests
import pandas as pd
import numpy as np
from zenml import step
from typing import List, Dict, Any

@step
def custom_mlflow_predictor(
    data: str,
    model_endpoint: str,
) -> List[float]:
    """Custom step to make predictions using the deployed MLflow model."""
    if model_endpoint.startswith("not_deployed") or model_endpoint.startswith("error"):
        raise ValueError(f"Model not properly deployed: {model_endpoint}")
        
    # Parse the JSON data
    parsed_data = json.loads(data)
    
    # Prepare features - use the same columns as in your predictor step
    columns = [
        "payment_sequential", "payment_installments", "payment_value", 
        "price", "freight_value", "product_name_lenght", 
        "product_description_lenght", "product_photos_qty", 
        "product_weight_g", "product_length_cm", 
        "product_height_cm", "product_width_cm"
    ]
    
    # Create DataFrame
    df = pd.DataFrame(parsed_data["data"], columns=columns)
    
    # Format for MLflow
    input_data = df.values.tolist()
    payload = json.dumps({"dataframe_split": {"columns": columns, "data": input_data}})
    
    # Try to connect to the server with retries
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            print(f"Sending prediction request to {model_endpoint} (attempt {attempt+1}/{max_retries})...")
            response = requests.post(
                model_endpoint,
                headers={"Content-Type": "application/json"},
                data=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                predictions = response.json().get('predictions', [])
                print(f"Received {len(predictions)} predictions")
                return predictions
            else:
                print(f"Error response: {response.status_code}, {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            
        # Wait before retrying
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2
    
    raise RuntimeError(f"Failed to get predictions after {max_retries} attempts")