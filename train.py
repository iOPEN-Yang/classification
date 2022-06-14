import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
import torch.optim as optim
from tqdm import tqdm

from utils.utils import *
from config import cfg
import predict



def main(config):
    device = torch.device(config.train_gpu if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    
    train_loader, validate_loader, train_num, val_num = loading_data(mode="train", config=config)
    
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
    
    # 加载预训练模型
    if config.pretrained is True:
        model_weight_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "pre_pth", config.model_weight_path)
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
        # for param in net.parameters():
        #     param.requires_grad = False

    # change fc layer structure
    if config.network in ["AlexNet", "VGG11", 'VGG13', "VGG16", 'VGG19', 'MobileNetv2', 'MobileNetv3', 'MobileNetv3_large']:
        in_channel = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_channel, 6)
    elif config.network in ['GoogleNet', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']:
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, 6)
        
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = config.epochs
    best_acc = 0.0
    model_name = config.network
    if config.pretrained is True:
        save_path = 'checkpoint/{}_yes.pth'.format(model_name)
    else:
        save_path = 'checkpoint/{}_no.pth'.format(model_name)
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), os.path.join(base_dir, save_path))
        
        logger_txt("train", config, [epoch+1, running_loss / train_steps, val_accurate, best_acc])

    print('Finished Training')


if __name__ == '__main__':
    main(cfg)
    predict.main(cfg)
