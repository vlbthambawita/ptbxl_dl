import lightning.pytorch as pl
# main.py
from lightning.pytorch.cli import LightningCLI
from data import PTBXLDataModule 
from models import CNN
import torch.nn as nn
import torch.optim as optim

# simple demo classes for your convenience
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


class ECGModel(pl.LightningModule):

    def __init__(self,  
                optimizer_name = "Adam",
                optimizer_hparams = {"lr":0.0001}
                ):
        
        super().__init__()
    
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()

        self.loss_module = nn.BCELoss() #nn.BCEWithLogitsLoss()
         
         # Define the model
        hidden_dim = 32
        dropout = 0.2
        self.model = CNN(12, 5, hidden_dim, dropout)


    def training_step(self, batch, batch_idx):
        ecg = batch["ecg"]
        gt = batch["class"]
        pred = self.model(ecg)
        loss = self.loss_module(pred, gt)

        #print("gt=", gt)
        #print("pred=", pred)

        acc = ((pred > 0.5) == gt).float().mean()
        #print(ecg.shape)
        #print("gt_shape=", gt.shape)
        #print(((pred > 0.5) == gt))
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ecg = batch["ecg"]
        gt = batch["class"]
        pred = self.model(ecg)
        loss = self.loss_module(pred, gt)

        
        acc = ((pred > 0.5) == gt).float().mean()
        #print(ecg.shape)
        #print("gt_shape=", gt.shape)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):

        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.Adam(self.parameters(), **self.hparams.optimizer_hparams)
        return [optimizer], []

       


def cli_main():

    cli = LightningCLI(ECGModel, PTBXLDataModule,
                        save_config_kwargs={"config_filename": "test_config.yaml", 'overwrite':True})


if __name__ == "__main__":
    cli_main()


    



