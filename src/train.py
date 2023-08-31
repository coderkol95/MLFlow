import pytorch_lightning as pl
import mlflow.pytorch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from data_work import data_module
from neural_network import regresion_network
# from pyngrok import ngrok,conf
import torch
import numpy as np
import os
import argparse
from datetime import datetime

class log_losses(Callback):

    def on_train_epoch_end(self, trainer, pl_module):
        mlflow.log_metric('train_loss_epochs', trainer.logged_metrics['train_loss'])
    def on_validation_epoch_end(self, trainer, pl_module):
        mlflow.log_metric('val_loss_epochs', trainer.logged_metrics['val_loss'])
    def on_test_epoch_end(self, trainer, pl_module):
        mlflow.log_metric('test_loss_epochs', trainer.logged_metrics['test_loss'])

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def train(epochs,lr):
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    checkpoint = ModelCheckpoint(monitor="val_loss")
    callbacks=[early_stop,checkpoint, log_losses()]
    set_seed()
    trainer=pl.Trainer(max_epochs=epochs, enable_progress_bar=True, callbacks=callbacks)
    data=data_module()
    model=regresion_network(lr)

    with mlflow.start_run() as run:
        # mlflow.log_input(mlflow.data.from_numpy(features=X_train,targets=y_train,source="sklearn/data/trainset.csv"), context='train')
        # mlflow.log_input(mlflow.data.from_numpy(features=X_test,targets=y_test,source="sklearn/data/testset.csv"), context='test')

        mlflow.pytorch.autolog()
        trainer.fit(model=model,datamodule=data)
        metrics=trainer.logged_metrics
        data_to_log = {"date":str(datetime.today().date()),"runID": [run.info.run_id],"train_loss":metrics["train_loss"].numpy(),"val_loss":metrics["val_loss"].numpy()}
        trainer.test(model=model,datamodule=data)
        metrics=trainer.logged_metrics
        data_to_log.update({"test_loss":metrics["test_loss"].numpy()})
        print(data_to_log)
        mlflow.log_table(data=data_to_log, artifact_file="comparison_table.json")

if __name__=="__main__":

    a=argparse.ArgumentParser()

    a.add_argument("epochs",default=3, type=int)
    a.add_argument("lr",default=0.05, type=float)

    args=a.parse_args()
    epochs=args.epochs
    lr=args.lr

    train(epochs,lr)