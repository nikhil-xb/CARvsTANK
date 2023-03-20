import numpy as np
import pandas as pd
import gc
import torch_xla
from torch.utils.data import Dataset, DataLoader
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from sklearn.model_selection import train_test_split
from src.dataset import CarTank
from src.config import config
from src.augmentations import Augments
from src.data import data
from src.model import *
from src.trainer import Trainer
from datetime import datetime

def fit_tpu(Trainer,device,epochs,train_loader,valid_loader):
    valid_loss_min = np.Inf  # track change in validation loss

    # keeping track of losses as it happen
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    for epoch in range(1, epochs + 1):
        gc.collect()
        para_train_loader = pl.ParallelLoader(train_loader, [device])
        xm.master_print(f"{'='*50}")
        xm.master_print(f"EPOCH {epoch} - TRAINING...")
        train_loss, train_acc = Trainer.train_one_cycle(
            para_train_loader.per_device_loader(device))
        xm.master_print(
            f"\n\t[TRAIN] EPOCH {epoch} - LOSS: {train_loss}, ACCURACY: {train_acc}\n"
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        gc.collect()

        if valid_loader is not None:
            gc.collect()
            para_valid_loader = pl.ParallelLoader(valid_loader, [device])
            xm.master_print(f"EPOCH {epoch} - VALIDATING...")
            valid_loss, valid_acc,op_model = Trainer.valid_one_cycle(
                para_valid_loader.per_device_loader(device))
            xm.master_print(f"\t[VALID] LOSS: {valid_loss}, ACCURACY: {valid_acc}\n")
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            gc.collect()

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min and epoch != 1:
                xm.master_print(
                    "Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...".format(
                        valid_loss_min, valid_loss
                    )
                )
                xm.save(op_model.state_dict(), 'modelv1.pth')
            valid_loss_min = valid_loss

    return {
        "train_loss": train_losses,
        "valid_losses": valid_losses,
        "train_acc": train_accs,
        "valid_acc": valid_accs,
        "Model": op_model
    }
def _run():
    train,test= data(config.test_path), data(config.train_path)
    train_df, valid_df= train_test_split(train,test_size=0.20,shuffle= True, random_state= 42,stratify= train.label.values )

    train_dataset = CarTank(config.train_path,train_df, augments=Augments.train)
    valid_dataset = CarTank(config.train_path,valid_df,augments=Augments.valid )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True,
        )

    valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False,
        )
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.BATCH,
            sampler=train_sampler,
            drop_last=True,
            num_workers=4,
        )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.BATCH,
        sampler=valid_sampler,
        drop_last=True,
        num_workers=4,
        )
    criterion= nn.CrossEntropyLoss()
    device= xm.xla_device()
    model= VITModel(num_classes=2,pretrained=True)
    model.to(device)
    lr= config.LR * xm.xrt_world_size()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    xm.master_print(f"INITIALIZING TRAINING ON {xm.xrt_world_size()} TPU CORES")
    start_time = datetime.now()
    xm.master_print(f"Start Time: {start_time}")
    trainer= Trainer(model,optimizer,criterion,device)
    logs= fit_tpu(Trainer=trainer,device= device,epochs= config.EPOCHS,train_loader=train_loader,valid_loader=valid_loader)
    xm.master_print(f"Execution time: {datetime.now() - start_time}")

    xm.master_print("Saving Model")
    xm.save(
        logs['Model'].state_dict(), f'model_5e_{datetime.now().strftime("%Y%m%d-%H%M")}.pth')
    

if __name__=='__main__':
   
    def _mp_fn(rank, flags):
        torch.set_default_tensor_type("torch.FloatTensor")
        a = _run()


    # _run()
    FLAGS = {}
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method="fork")

