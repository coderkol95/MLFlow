# Running training workflows:

    `mlflow run . -e train -P epochs=2 -P lr=0.1 --env-manager="local"`

# Running registration workflows:

    `mlflow run . -e register -P model_name="alpha" metric="val_loss" -P lookback_duration="1" --env-manager="local"`

# Running deployment workflows:

    `mlflow run . -e deploy -P model_name="alpha" -P model_version=1  -P model_stage="Staging" --env-manager="local"`