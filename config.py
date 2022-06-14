import os
from easydict import EasyDict as edict
import time
import torch

__C = edict()
cfg = __C

__C.network = 'MobileNetv3_large'  # AlexNet, VGG11, VGG13, VGG16, VGG19, GoogleNet,
                     # ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
                     # MobileNetv2, MobileNetv3, MobileNetv3_large
__C.pretrained = True  # True, False 
__C.train_gpu = "cuda:1"
__C.dataset_dir = r"/data1/yanghao/pythonProject/homework/data_set/plane_data"





if __C.pretrained == False:
    __C.epochs = 100
else:
    __C.epochs = 30

if __C.network == 'AlexNet':
    __C.model_weight_path = "alexnet-pre.pth"
    __C.batch_size = 32
elif __C.network == 'VGG11':
    __C.model_weight_path = "vgg11-8a719046.pth"
    __C.batch_size = 32
elif __C.network == 'VGG13':
    __C.model_weight_path = "vgg13-19584684.pth"
    __C.batch_size = 32
elif __C.network == 'VGG16':
    __C.model_weight_path = "vgg16-397923af.pth"
    __C.batch_size = 32
elif __C.network == 'VGG19':
    __C.model_weight_path = "vgg19-dcbb9e9d.pth"
    __C.batch_size = 32
elif __C.network == 'GoogleNet':
    __C.model_weight_path = "googlenet-1378be20.pth"
    __C.batch_size = 32
elif __C.network == 'ResNet18':
    __C.model_weight_path = "resnet18-f37072fd.pth"
    __C.batch_size = 32
elif __C.network == 'ResNet34':
    __C.model_weight_path = "resnet34-pre.pth"
    __C.batch_size = 32
elif __C.network == 'ResNet50':
    __C.model_weight_path = "resnet50-0676ba61.pth"
    __C.batch_size = 32
elif __C.network == 'ResNet101':
    __C.model_weight_path = "resnet101-63fe2227.pth"
    __C.batch_size = 32
elif __C.network == 'ResNet152':
    __C.model_weight_path = "resnet152-394f9c45.pth"
    __C.batch_size = 32
elif __C.network == 'MobileNetv2':
    __C.model_weight_path = "mobilenet_v2-b0353104.pth"
    __C.batch_size = 64
elif __C.network == 'MobileNetv3':
    __C.model_weight_path = "mobilenet_v3_small-047dcff4.pth"
    __C.batch_size = 64
elif __C.network == 'MobileNetv3_large':
    __C.model_weight_path = "mobilenet_v3_large-8738ca79.pth"
    __C.batch_size = 64