from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import os
import pandas as pd
import numpy as np

from .config import config
from .augmentations import Augments

class CarTank(Dataset):
    def __init__(self,path,df,augments):
        super().__init__()
        self.path=path
        self.data= df.values
        self.augments= augments
    def __getitem__(self,idx):
        img_id,label= self.data[idx]
        img= cv2.imread(self.path+img_id)
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.augments is not None:
            aug= self.augments(image=img)
            img= aug['image']
#         img= np.moveaxis(img,2,0)
        return img, label
                                                                  
    def __len__(self):
        return len(self.data)
            