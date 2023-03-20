import timm
import torch.nn as nn
import torch

#Model Class

class ResNetModel(nn.Module):
    def __init__(self, num_classes=2, model_name= 'resnet50', pretrained= True):
        super(ResNetModel,self).__init__()
        self.model= timm.create_model(model_name,pretrained= pretrained)
        self.model.fc= nn.Sequential(
            nn.Linear(self.model.fc.in_features,num_classes,bias=False))
    def forward(self,x):
        x= self.model(x)
        return x
class VITModel(nn.Module):
    def __init__(self,num_classes=2,model_name='vit_base_patch16_224',pretrained=True):
        super(VITModel,self).__init__()
        self.model= timm.create_model(model_name,pretrained=pretrained)
        # if pretrained:
        #     self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.head= nn.Linear(self.model.head.in_features, num_classes)
    def forward(self,x):
        x= self.model(x)
        return x