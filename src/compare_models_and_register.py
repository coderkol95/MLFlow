import mlflow
import json

table=mlflow.load_table(artifact_file='comparison_table.json')
table.sort_values(['test_loss'], ascending=False, inplace=True)
print(table.head())

top_model_run_ID=table.head(1)['runID'].values[0]

mlflow.register_model(
    f"runs:/{top_model_run_ID}/model", "regression_NN_model",
    )