import lightning.pytorch as pl
# main.py
from lightning.pytorch.cli import LightningCLI
from data import PTBXLDataModule 
from models import CNN
import torch.nn as nn
import torch.optim as optim

# simple demo classes for your convenience
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule
from pytorch_lightning.loggers import WandbLogger


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

        #self.training_step_outputs = []


    def training_step(self, batch, batch_idx):
        ecg = batch["ecg"]
        gt = batch["class"]
        pred = self.model(ecg)
        loss = self.loss_module(pred, gt)

        #print("gt=", gt)
        #print("pred=", pred)
       # self.training_step_outputs.append(preds)

        acc = ((pred > 0.5) == gt).float().mean()
        #print(ecg.shape)
        #print("gt_shape=", gt.shape)
        #print(((pred > 0.5) == gt))
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True, logger=True)
        self.log("train_acc_step", acc)
        return loss

    def on_train_epoch_end(self):
        #all_preds = torch.stack(self.training_step_outputs)
        #print("all_preds=", outputs.shape)
        pass


    def validation_step(self, batch, batch_idx):
        ecg = batch["ecg"]
        gt = batch["class"]
        pred = self.model(ecg)
        loss = self.loss_module(pred, gt)

        
        acc = ((pred > 0.5) == gt).float().mean()
        #print(ecg.shape)
        #print("gt_shape=", gt.shape)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True, logger=True)
        
    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):

        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.Adam(self.parameters(), **self.hparams.optimizer_hparams)
        return [optimizer], []

       

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--wandb_name", default="")
        parser.add_argument("--wandb_entity", default="simulamet_mlc")
        parser.add_argument("--wandb_project", default="PTBXL_ECG")
        parser.add_argument("--output_dir", default="output/new")
        #parser.set_defaults({"model.output_dir": "output/test/"})
        
        #parser.set_defaults({"model.config"})
        #parser.link_arguments("trainer.callbacks[" + str(0) +"]", "model.output_dir")
        #parser.link_arguments("output_dir", "model.output_dir")
        #parser.link_arguments("wandb_name", "model.wandb_name")
        
    #def add_default_arguments_to_parser(self, parser):
        
        
        
    def instantiate_classes(self):
        #print(self.config[self.config.subcommand])
        
        # Call to the logger before initiate other clases, because Trainer class init logger if we didn´t do it
        logger = WandbLogger(entity=self.config[self.config.subcommand].wandb_entity, 
                             project=self.config[self.config.subcommand].wandb_project,
                             name=self.config[self.config.subcommand].wandb_name)
        super().instantiate_classes() # call to super class instatiate_classes()


def cli_main():

    cli = MyLightningCLI(ECGModel, PTBXLDataModule,
                        save_config_kwargs={"config_filename": "test_config.yaml", 'overwrite':True})


if __name__ == "__main__":
    cli_main()


    



