import pytorch_lightning as pl
import torch
from torch.nn.functional import relu, mse_loss, leaky_relu, elu
from torch.optim import Adam, RMSprop

class regresion_network(pl.LightningModule):

    def __init__(self):
        super(regresion_network,self).__init__()        
        self.loss=mse_loss
        self.inp=torch.nn.Linear(in_features=5,out_features=10)
        # self.h1=torch.nn.Linear(in_features=10,out_features=10)
        self.out=torch.nn.Linear(in_features=10,out_features=1)

    def configure_optimizers(self):
        return Adam(params=self.parameters(), lr=0.02)
      
    def forward(self, X):
        
        self.X = relu(self.inp(X))
        # self.X = relu(self.h1(self.X))
        self.X = relu(self.out(self.X))
        return relu(self.X)
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch 
        logits = self.forward(x)
        loss = self.loss(logits, y) 
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, valid_batch, batch_idx): 
        x, y = valid_batch 
        logits = self.forward(x) 
        loss = self.loss(logits, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, test_batch, batch_idx): 
        x, y = test_batch 
        logits = self.forward(x) 
        loss = self.loss(logits, y)
        self.log("test_loss", loss)
        return loss
