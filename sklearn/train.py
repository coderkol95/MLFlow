import warnings
import argparse
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LinearRegression
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import json
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from urllib.parse import urlparse

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

with open('sklearn/config.json','r') as f:
    config = json.load(f)
RANDOM_STATE=config['RANDOM STATE']

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

### Data work
def read_data():
    X=np.loadtxt("sklearn/data/X.csv", dtype=np.float32, delimiter=',')
    y=np.loadtxt("sklearn/data/y.csv", dtype=np.float32, delimiter=',')
    return X,y

def get_data_pipe():

    """
    It is assumed here that the data is numerical only. If not, a column transformer can be used here instead.
    """
    return Pipeline([('scaler',StandardScaler())])

def get_sklearn_model():
    """
    Return the model, its hyperparams and their default values
    """
    return ElasticNet(random_state=RANDOM_STATE), ['alpha','l1_ratio'], [0.5,0.5]    

if __name__ == "__main__":

    data_pipe=get_data_pipe()
    model_pipe,model_params,default_param_values=get_sklearn_model()

    X,y=read_data()
    X_train, X_test,y_train,y_test = train_test_split(X,y,random_state=RANDOM_STATE)
    trainset=np.c_[X_train,y_train]
    testset=np.c_[X_test,y_test]

    np.savetxt(fname="sklearn/data/trainset.csv",X=trainset, delimiter=',')
    np.savetxt(fname="sklearn/data/testset.csv",X=trainset, delimiter=',')

    model_args={'model__l1_ratio':0.01, 'model__alpha':0.05}

    pipe = Pipeline([('data_pipe',data_pipe),('model',model_pipe)])

    with mlflow.start_run():
        mlflow.log_input(mlflow.data.from_numpy(features=X_train,targets=y_train,source="sklearn/data/trainset.csv"), context='train')
        mlflow.log_input(mlflow.data.from_numpy(features=X_test,targets=y_test,source="sklearn/data/testset.csv"), context='test')
        mlflow.set_tag("release","training")
        pipe.set_params(**model_args)
        pipe.fit(X_train, y_train)

        predicted_qualities = pipe.predict(X_test)

        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        mlflow.log_params(model_args)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = pipe.predict(X_train)
        signature = infer_signature(X_train, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                pipe, "model", registered_model_name="ElasticnetWineModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(pipe, "model", signature=signature)