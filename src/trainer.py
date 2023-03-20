import torch
import tqdm
import torch.nn as nn
from torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
import numpy as np
import timm
from torch.utils.data import DataLoader, Dataset
from .config import config
class Trainer:
    def __init__(self, model,optimizer,criterion,device): #,train_dataloader,valid_dataloader):
        self.model= model
        self.criterion= criterion
        self.optimizer= optimizer
        self.device=device
       # self.train= train_dataloader
       # self.valid= valid_dataloader
    def train_one_cycle(self,train_dataloader): #
        self.model.train()
        train_prog_bar= tqdm.tqdm(train_dataloader,total= len(train_dataloader)) #train_dataloader
        epoch_loss, epoch_accuracy= 0,0
        for i,data in enumerate(train_prog_bar):
            if self.device.type=="cuda":
                xtrain,ytrain= data[0].cuda(), data[1].cuda()
            elif self.device.type == "xla":
                xtrain= data[0].to(self.device,dtype=torch.float32)
                ytrain= data[1].to(self.device,dtype=torch.int64)
            self.optimizer.zero_grad()
            z= self.model(xtrain)
            train_loss= self.criterion(z,ytrain)
            train_loss.backward()
            accuracy= (z.argmax(dim=1)==ytrain).float().mean()
            epoch_loss+= train_loss
            epoch_accuracy+= accuracy

            if self.device.type=='xla':
                xm.optimizer_step(self.optimizer)

                if i%20==0:
                    xm.master_print(f'\tBATCH{i+1}/{len(train_dataloader)} - LOSS:{train_loss}')
            else:
                self.optimizer.step()
        return epoch_loss/len(train_dataloader), epoch_accuracy/len(train_dataloader) #train_dataloader
                
    def valid_one_cycle(self,valid_dataloader): #
        
        valid_loss, valid_accuracy= 0.0, 0.0
        self.model.eval()
        valid_prog_bar= tqdm.tqdm(valid_dataloader,total= len(valid_dataloader)) #valid_dataloader
        for i,data in enumerate(valid_prog_bar):
            if self.device.type=="cuda":
                xval= data[0].cuda()
                yval=  data[1].cuda()
            elif self.device.type=="xla":
                xval= data[0].to(self.device, dtype=torch.float32)
                yval= data[1].to(self.device, dtype=torch.int64)
            with torch.no_grad():
    
                val_z= self.model(xval)
                loss= self.criterion(val_z,yval)
                accuracy= (val_z.argmax(dim=1)==yval).float().mean()

                valid_loss+= loss
                valid_accuracy+= accuracy
        return valid_loss/len(valid_dataloader), valid_accuracy/len(valid_dataloader),self.model # valid_dataloader
        