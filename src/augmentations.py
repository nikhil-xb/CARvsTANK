from .config import config
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
class Augments:
    train= A.Compose([
        A.Resize(config.IMG_SIZE,config.IMG_SIZE,interpolation= cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomResizedCrop(config.IMG_SIZE,config.IMG_SIZE,p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
        ToTensorV2(),
    ],p=1)
    valid= A.Compose([
        A.Resize(config.IMG_SIZE,config.IMG_SIZE,interpolation= cv2.INTER_NEAREST),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
        ToTensorV2()
    ],p=1)