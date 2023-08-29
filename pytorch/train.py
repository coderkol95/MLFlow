import json
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import mlflow.pytorch
from mlflow import log_params
from torch.optim import Adam, RMSprop
from torch.nn.functional import relu, mse_loss, leaky_relu, elu
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

with open('config.json','r') as f:
    config = json.load(f)
BATCH_SIZE=config['BATCH SIZE']

class dataloader(pl.LightningDataModule):

    def __init__(self):
        super(dataloader,self).__init__()
  
    def setup(self,stage=None):
        if stage=='fit':
            X=np.loadtxt("data/X.csv", dtype=np.int8, delimiter=',')
            y=np.loadtxt("data/y.csv", dtype=np.int8, delimiter=',')
            indices=np.arange(X.shape[0])
            train_indices=np.random.choice(indices,int(0.8*X.shape[0]))
            indices=list(set(indices).difference(set(train_indices)))
            val_indices=np.random.choice(indices,int(0.1*X.shape[0]))
            test_indices=list(set(indices).difference(set(val_indices)))

            self.X_train,self.y_train=X[train_indices],y[train_indices]
            self.X_val,self.y_val=X[val_indices],y[val_indices]
            self.X_test,self.y_test=X[test_indices],y[test_indices]

    def train_dataloader(self):        
        return DataLoader(TensorDataset(torch.from_numpy(self.X_train),torch.from_numpy(self.y_train)), batch_size=BATCH_SIZE, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(TensorDataset(torch.from_numpy(self.X_val),torch.from_numpy(self.y_val)), batch_size=BATCH_SIZE, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(TensorDataset(torch.from_numpy(self.X_test),torch.from_numpy(self.y_test)), batch_size=BATCH_SIZE, shuffle=False)
    
class regresion_network(pl.LightningModule):

    def __init__(self):
    
        super(regresion_network,self).__init__()        
        self.loss=mse_loss

        self.inp=torch.nn.Linear(in_features=5,out_features=10)
        self.h1=torch.nn.Linear(in_features=10,out_features=10)
        self.out=torch.nn.Linear(in_features=10,out_features=1)

    def configure_optimizers(self):
        return Adam(params=self.parameters())
      
    def forward(self, X):
        
        self.X = relu(self.inp(X))
        self.X = relu(self.h1(X))
        self.X = relu(self.out(X))
        return relu(self.X)
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch 
        logits = self.forward(x.type(torch.float32))
        print("Prediction shape",logits.shape)
        print("Target shape",y.shape)
        loss = self.loss(logits.float(), y.float()) 
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, valid_batch, batch_idx): 
        x, y = valid_batch 
        logits = self.forward(x.type(torch.float32)) 
        loss = self.loss(logits.float(), y.float())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, test_batch, batch_idx): 
        x, y = test_batch 
        logits = self.forward(x.type(torch.float32)) 
        loss = self.loss(logits.float(), y.float())
        self.log("test_loss", loss)
        return loss

def train(args):

    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    checkpoint = ModelCheckpoint(monitor="val_loss")
    callbacks=[early_stop,checkpoint]
    trainer=pl.Trainer(max_epochs=1, enable_progress_bar=True, callbacks=callbacks)
    mlflow.pytorch.autolog()
    data=dataloader()
    model=regresion_network()
    mlflow.pytorch.autolog()

    with mlflow.start_run() as run:
        trainer.fit(model=model, datamodule=data)

    trainer.test(model=model, datamodule=data)

    return model

if __name__=="__main__":

    train(None)