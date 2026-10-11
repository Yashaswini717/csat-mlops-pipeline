@echo off
echo Starting MLflow server for model: model
echo Model URI: runs:/8c84230f71ff4d30b9be51a7469f8a21/model
mlflow models serve -m "runs:/8c84230f71ff4d30b9be51a7469f8a21/model" --no-conda -p 5000
pause
