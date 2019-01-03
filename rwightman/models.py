from torch import nn
from pretrainedmodels.models import bninception, resnet101

def get_net(model_name,  
            num_classes,
            drop_rate,
            channels):
    if model_name=='bninception':
        model = bninception(pretrained="imagenet")
        model.global_pool = nn.AdaptiveAvgPool2d(1)
        model.conv1_7x7_s2 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        model.last_linear = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Dropout(drop_rate),
                nn.Linear(1024, num_classes))
    elif model_name=='resnet101':
        model = resnet101(pretrained="imagenet")
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.conv1 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        model.last_linear = nn.Sequential(
                nn.BatchNorm1d(2048),
                nn.Dropout(drop_rate),
                nn.Linear(2048, num_classes))
    else:
        assert False and "Invalid model"   
    return model