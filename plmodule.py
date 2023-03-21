import lightning.pytorch as pl
from models import CNN
import torch.nn as nn
import torch.optim as optim
import torch





class ECGModel(pl.LightningModule):

    def __init__(self,  
                optimizer_name = "Adam",
                optimizer_hparams = {"lr":0.0001},
                lr_scheduler_hparams = {"step_size": 1},
                prediction_threshold = 0.5,
                hidden_dim = 32,
                dropout = 0.2
                ):
        
        super().__init__()
    
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        self.prediction_threshold = prediction_threshold

        self.loss_module = nn.BCELoss() #nn.BCEWithLogitsLoss()
         
         # Define the model
        #hidden_dim = 32
        #dropout = 0.2
        self.model = CNN(12, 5, hidden_dim, dropout)

        #self.training_step_outputs = []
        self.validation_step_outputs_acc = []


    def training_step(self, batch, batch_idx):
        ecg = batch["ecg"]
        gt = batch["class"]
        pred = self.model(ecg)
        loss = self.loss_module(pred, gt)

        #print("gt=", gt)
        #print("pred=", pred)
       # self.training_step_outputs.append(preds)

        acc = ((pred > self.prediction_threshold) == gt).float().mean()
        #print(ecg.shape)
        #print("gt_shape=", gt.shape)
        #print(((pred > 0.5) == gt))
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True, logger=True)
        #self.log("train_acc_step", acc)
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
        self.validation_step_outputs_acc.append(acc)

    def on_validation_epoch_end(self):
        all_acc = torch.stack(self.validation_step_outputs_acc)
        #print(all_acc.shape)
        self.log("val_epoch_acc", all_acc.mean())

        self.validation_step_outputs_acc.clear()
        
    def test_step(self, batch, batch_idx):
        ecg = batch["ecg"]
        gt = batch["class"]
        pred = self.model(ecg)

    def configure_optimizers(self):

        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.Adam(self.parameters(), **self.hparams.optimizer_hparams)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.hparams.lr_scheduler_hparams)
        return [optimizer], [lr_scheduler]
