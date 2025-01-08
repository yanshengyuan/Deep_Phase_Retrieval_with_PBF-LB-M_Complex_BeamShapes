import random, os
import pandas as pd
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import mlflow
import time
from torchmetrics.functional.regression import mean_absolute_percentage_error

from architecture_1_resnet import ResNet, BasicBlock
from architecture_2_smnet import Net as smNet
from architecture_3_phasenet import Net as phaseNet
from model import ResNet18, FC
from scaling_model import FC3, FC6, FC18, FC24
from scaling_model import ResNet10, ResNet34, ResNet50, ResNet101

import timm
import os
from dataset import DataLoaderTrain, DataLoaderVal

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from configs_vit_training_time import *

torch.manual_seed(42)
seed = experiment_params['seed']
doi = experiment_params['doi']
arch = experiment_params['arch']
foc_list = experiment_params['focal_dist']
InputPlanes = len(foc_list)

run_name = experiment_params['run_name']
data_folder = experiment_params['data_folder']
init_lr=experiment_params['init_lr']
stepsize=experiment_params['stepsize']
modelsize=experiment_params['modelsize']
num_training_samples=experiment_params['num_samples']
num_z=experiment_params['num_zernikes']

random.seed(seed)
torch.manual_seed(seed)

start_time=time.time()

class PlModel(pl.LightningModule):
    def __init__(self, model):
        super(PlModel, self).__init__()
        self.model = model

    def configure_optimizers(self):
        #resnet
        if doi == '10.1063/1.5125252':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=init_lr, betas=(.9, .999), eps=1e-8)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-6, eps=1e-4)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'ARE'}
        
        #smnet
        elif doi == '10.1038/s41592-018-0153-5':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=init_lr)
            return optimizer

        #phasenet
        elif doi == '10.1364/OE.401933':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=init_lr)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize)
            return [optimizer], [lr_scheduler]

        else: # deliverable
            optimizer = torch.optim.Adam(self.model.parameters(), lr=init_lr, weight_decay=1e-4)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=0.5)
            return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        y, x = batch
        y_hat = self.model(x)

        #mape = 0
        #for i in range(len(x)):
        #    mape += mean_absolute_percentage_error(y_hat[i], y[i])
        #mape /= len(x)
        mape2 = mean_absolute_percentage_error(y_hat, y)
        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = F.l1_loss(y_hat, y)

        #self.log('ARE', mape, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log('ARE', mape2, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log('mse', mse_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log('mae', mae_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return mse_loss

    def validation_step(self, batch, batch_idx):
        y, x = batch
        y_hat = self.model(x)

        #mape = 0
        #for i in range(len(x)):
        #    mape += mean_absolute_percentage_error(y_hat[i], y[i])
        #mape /= len(x)
        mape2 = mean_absolute_percentage_error(y_hat, y)
        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = F.l1_loss(y_hat, y)

        #self.log('ARE', mape, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log('ARE_val', mape2, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log('mse_val', mse_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log('mae_val', mae_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)

    def forward(self, x):
        return self.model(x)



if doi == '10.1063/1.5125252':
    model = PlModel(ResNet(num_z, BasicBlock, [2, 2, 2]))  # paper-1
elif doi == '10.1038/s41592-018-0153-5':
    model = PlModel(smNet(3, num_z)) # paper-2
elif doi == '10.1364/OE.401933':
    model = PlModel(phaseNet(3, num_z)) # paper-3
else:
    if arch=='resnet':
        if modelsize=='base':
            model = PlModel(ResNet18(num_z, InputPlanes))
        elif modelsize=='tiny':
            model = PlModel(ResNet10(num_z))
        elif modelsize=='small':
            model = PlModel(ResNet34(num_z))
        elif modelsize=='large':
            model = PlModel(ResNet50(num_z))
        elif modelsize=='huge':
            model = PlModel(ResNet101(num_z))
    elif arch=='mlp':
        if modelsize=='base':
            model = PlModel(FC(num_z, InputPlanes))
        elif modelsize=='tiny':
            model = PlModel(FC3(num_z))
        elif modelsize=='small':
            model = PlModel(FC6(num_z))
        elif modelsize=='large':
            model = PlModel(FC18(num_z))
        elif modelsize=='huge':
            model = PlModel(FC24(num_z))
    elif arch=='ViT':
        if modelsize=='base':
            model = PlModel(timm.create_model('vit_base_patch16_224', pretrained=False, img_size=64,
                                              num_classes=num_z, in_chans=InputPlanes))
        elif modelsize=='tiny':
            model = PlModel(timm.create_model('vit_tiny_patch16_224', pretrained=False, img_size=64,
                                              num_classes=num_z))
        elif modelsize=='small':
            model = PlModel(timm.create_model('vit_small_patch16_224', pretrained=False, img_size=64,
                                              num_classes=num_z))
        elif modelsize=='large':
            model = PlModel(timm.create_model('vit_large_patch16_224', pretrained=False, img_size=64,
                                              num_classes=num_z))
        elif modelsize=='huge':
            model = PlModel(timm.create_model('vit_huge_patch14_224_in21k', pretrained=False, img_size=64,
                                              num_classes=num_z))
# Load datasets
dataset_train = DataLoaderTrain(data_folder, foc_list, num_training_samples, num_z)
dataset_val = DataLoaderVal(data_folder, foc_list, num_z)

print(dataset_train.__len__())
print(dataset_val.__len__())

dl_train = torch.utils.data.DataLoader(dataset_train, 
                                       batch_size=experiment_params['batch_size'], 
                                       num_workers=experiment_params['num_workers'], 
                                       shuffle=False)
dl_val = torch.utils.data.DataLoader(dataset_val, 
                                     batch_size=experiment_params['batch_size'], 
                                     num_workers=experiment_params['num_workers'], 
                                     shuffle=False)

checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=experiment_params['save_top_k'], monitor=experiment_params['best_model_metric'])
trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=experiment_params['num_epochs'], callbacks=[checkpoint_callback], log_every_n_steps=experiment_params['log_every_n_steps'])

EXPERIMENT_NAME = experiment_params['experiment_name']  # shown as exp-ID(numbers) in "mlruns" folder:
                              # can be obtained by: MlflowClient().get_experiment(exp-ID)
                              # that folder includes runs with different IDs


# Set given experiment as active experiment. If experiment does not exist, create an experiment with provided name.
# mlflow.set_experiment(experiment_id='388945639541205512')
mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)


# Auto log all MLflow entities
mlflow.pytorch.autolog(exclusive=False)


# Train the model
with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_param('DOI', doi)
    mlflow.log_param('seed', seed)
    trainer.fit(model, dl_train, dl_val)

end_time=time.time()
total_time=(end_time - start_time) / 60
print(f"Total time training: {total_time:.2f} min")