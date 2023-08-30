from mlflow import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="regression_NN_model", version=1, stage="Staging"
)