import os
import subprocess
import time
import json
import requests
import pandas as pd

# Path to your logged model - update this to your latest model path
model_path = "C:\\Users\\yashu\\AppData\\Roaming\\zenml\\local_stores\\837c56be-36be-4929-8b33-657a99e4924a\\mlruns\\137236167669405430\\61474e7576b04e7ea46bcfc048c26d0c\\artifacts\\model"

def start_mlflow_server(model_path):
    """Start MLflow server manually without daemon mode"""
    print("Starting MLflow server...")
    # Use subprocess.Popen to run non-blocking
    process = subprocess.Popen(
        f'mlflow models serve -m "{model_path}" --no-conda -p 5000',
        shell=True
    )
    print("Waiting for server to start (10 seconds)...")
    time.sleep(10)  # Give it more time to fully load the model
    return process

def test_prediction():
    """Make a test prediction against the server"""
    # Import the test data function
    from pipelines.utils import get_data_for_test
    
    # Get test data
    data = get_data_for_test()
    parsed_data = json.loads(data)
    
    # Prepare features - use the same columns as in your predictor step
    columns = [
        "payment_sequential", "payment_installments", "payment_value", 
        "price", "freight_value", "product_name_lenght", 
        "product_description_lenght", "product_photos_qty", 
        "product_weight_g", "product_length_cm", 
        "product_height_cm", "product_width_cm"
    ]
    
    df = pd.DataFrame(parsed_data["data"], columns=columns)
    
    # For MLflow format, convert to numpy list format
    input_data = df.values.tolist()
    
    # Try to send a prediction request
    try:
        print("Sending prediction request...")
        response = requests.post(
            "http://127.0.0.1:5000/invocations",
            headers={"Content-Type": "application/json"},
            # MLflow expects a specific format:
            data=json.dumps({"dataframe_split": {"columns": columns, "data": input_data}})
        )
        
        print("Status Code:", response.status_code)
        if response.status_code == 200:
            print("Predictions:", response.json())
            return True
        else:
            print("Error:", response.text)
            return False
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return False

if __name__ == "__main__":
    if not os.path.exists(model_path):
        print(f"Model path not found: {model_path}")
        print("Please update the model_path variable in this script with the correct path")
        exit(1)
    
    # Start the MLflow server
    server_process = start_mlflow_server(model_path)
    
    try:
        # Test prediction
        print("\nTesting prediction...")
        success = test_prediction()
        
        if success:
            print("\nSuccess! The MLflow server is working correctly.")
        else:
            print("\nPrediction test failed. The server might need more time to start or there might be a formatting issue.")
            print("Try again in a few seconds by pressing Enter...")
        
        print("\nMLflow server is running at http://127.0.0.1:5000/invocations (for POST requests only)")
        print("Press Enter to test again, or Ctrl+C to stop the server")
        
        # Keep the server running until user interrupts
        while True:
            input()  # Wait for Enter key
            test_prediction()
            
    except KeyboardInterrupt:
        print("\nShutting down MLflow server...")
        server_process.terminate()
        print("Done!")