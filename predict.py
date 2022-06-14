import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
import torch.optim as optim
from tqdm import tqdm

from utils.utils import *
from config import cfg


def main(config):
    device = torch.device(config.train_gpu if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    base_dir = os.path.abspath(os.path.dirname(__file__))

    test_loader, test_num = loading_data(mode="test", config=config)

    if config.network == 'AlexNet':
        net = models.alexnet()
    elif config.network == 'VGG11':
        net = models.vgg11()
    elif config.network == 'VGG13':
        net = models.vgg13()
    elif config.network == 'VGG16':
        net = models.vgg16()
    elif config.network == 'VGG19':
        net = models.vgg19()
    elif config.network == 'GoogleNet':
        net = models.googlenet()
    elif config.network == 'ResNet18':
        net = models.resnet18()
    elif config.network == 'ResNet34':
        net = models.resnet34()
    elif config.network == 'ResNet50':
        net = models.resnet50()
    elif config.network == 'ResNet101':
        net = models.resnet101()
    elif config.network == 'ResNet152':
        net = models.resnet152()
    elif config.network == 'MobileNetv2':
        net = models.mobilenet_v2()
    elif config.network == 'MobileNetv3':
        net = models.mobilenet_v3_small()
    elif config.network == 'MobileNetv3_large':
        net = models.mobilenet_v3_large()  
    
    # change fc layer structure
    if config.network in ["AlexNet", "VGG11", 'VGG13', "VGG16", 'VGG19', 'MobileNetv2', 'MobileNetv3', 'MobileNetv3_large']:
        in_channel = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_channel, 6)
    elif config.network in ['GoogleNet', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']:
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, 6)

    # 加载训练好的模型
    if config.pretrained is True:
        pth_name = 'checkpoint/{}_yes.pth'.format(config.network)
    else:
        pth_name = 'checkpoint/{}_no.pth'.format(config.network)
    model_weight_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), pth_name)
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    net.to(device)

    best_acc = 0.0

    # predict
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    y_true = np.array([])
    y_pred = np.array([])
    with torch.no_grad():
        val_bar = tqdm(test_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
            y_true = np.append(y_true, val_labels)
            y_pred = np.append(y_pred, (predict_y.to("cpu")).numpy())
    test_acc = acc / test_num

    print('test_accuracy: %.3f' % test_acc)

    logger_txt("test", config, [y_true, y_pred])

    print('Finished Predicting')


if __name__ == '__main__':
    main(cfg)
