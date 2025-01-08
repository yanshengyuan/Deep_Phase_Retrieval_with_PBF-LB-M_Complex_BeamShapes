import random, os
import pandas as pd
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import numpy as np
import mlflow
from dataset import DataLoaderTest

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from configs_test import *

bs=experiment_params['batch_size']
data_folder=experiment_params['data_folder']
foc_list=experiment_params['focal_dist']
model_path=experiment_params['model_path']

filename="chair-resnet"
val_dataset = DataLoaderTest(data_folder, foc_list)
dl_val = torch.utils.data.DataLoader(
    val_dataset, batch_size=bs, shuffle=False,
    num_workers=4, pin_memory=True, sampler=None)
print(val_dataset.__len__())

loaded_model = mlflow.pytorch.load_model(model_path).eval().cuda(device=0)
pred=[]
mae=0.0
for i, (target, inp) in enumerate(dl_val):
    inp = inp.cuda(device=0)
    target = target.cuda(device=0)
    zpred = loaded_model(inp)
    zpred = zpred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    for j in range(len(zpred)):
        pred.append(zpred[j])
        print(i*bs+j)
        print(zpred[j])
        print(target[j])
        
        mae_zk=0.0
        for k in range(len(zpred[j])):
            mae_zk += abs(zpred[j][k]-target[j][k])
        mae_zk /= len(zpred[j])
        mae += mae_zk
pred=np.array(pred)
np.save("./predictions/" + filename + ".npy", pred)

mae /= val_dataset.__len__()
print('\n MAE(Z): ', mae)