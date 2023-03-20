import os
train_path= '/kaggle/input/cars-and-tanks-image-classification/cars_tanks/train/'
test_path= '/kaggle/input/cars-and-tanks-image-classification/cars_tanks/test/'
MODEL_PATH= '/kaggle/input/vit-base-models-pretrained-pytorch/jx_vit_base_p16_224-80ecf9dd.pth'
EPOCHS= 10
n_cpu= os.cpu_count()
BATCH= 32
IMG_SIZE= 224
LR=1e-05
