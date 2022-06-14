"""
code from iopen.yang
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json

import json
import pickle
import random

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score, precision_score, recall_score, f1_score

from torchvision import transforms, datasets
import torch

import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


def P_R_A_F1(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    con_matrix = confusion_matrix(y_true, y_pred)
    return acc, precision, recall, f1, con_matrix


def logger_txt(mode, config, others):
    base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../log")

    if config.pretrained is True:
        logger_txt_name = "{}_yes.txt".format(config.network)
    elif config.pretrained is False:
        logger_txt_name = "{}_no.txt".format(config.network)

    if mode == "train":
        epoch, train_loss, val_acc, best_acc = others
        with open(os.path.join(base_dir, logger_txt_name), 'a') as f:
            f.write("[epoch {}]: train_loss={:.3f} val_acc={:.3f} best_acc={:.3f}\n".format(epoch, train_loss, val_acc,
                                                                                            best_acc))
    elif mode == "test":
        y_true, y_pred = others
        acc, precision, recall, f1, con_matrix = P_R_A_F1(y_true, y_pred)
        print(con_matrix)
        with open(os.path.join(base_dir, logger_txt_name), 'a') as f:
            f.write("\n\n")
            f.write("test_acc = {:.3f}\n".format(acc))
            f.write("precision = {:.3f}\n".format(precision))
            f.write("recall = {:.3f}\n".format(recall))
            f.write("f1_score = {:.3f}\n".format(f1))


def getFilePathList(path, filetype: str = 'txt'):
    """
    获取path路径下所有类型为filetype的文件路径
    :param path:
    :param filetype:
    :return:
    """
    pathList = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(filetype):
                pathList.append(os.path.join(root, file))
    return pathList  # 输出以filetype为后缀的列表


def getDirPathList(path):
    """
    获取文件夹下所有文件夹名称
    :param path:
    :return:
    """
    pathList = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            pathList.append(os.path.join(root, dir))
    return pathList


def read_test_data(root: str):
    assert os.path.exists(root), "test data root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    # class_indices = dict((k, v) for v, k in enumerate(flower_class))
    class_indices = {"negative": 0, "positive": 1}

    test_path = []
    test_label = []
    every_class_num = []

    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        # 获取该类别对应的索引
        image_class = class_indices[cla]
        if image_class == 1:
            assert image_class == 1
        # 记录该类别的样本数量
        every_class_num.append(len(images))

        for img_path in images:
            test_path.append(img_path)
            test_label.append(image_class)

    # print("{} images were found in the test dataset.".format(sum(every_class_num)))
    # print("{} images for test.".format(len(test_path)))

    return test_path, test_label


def indicators(y_true, y_pred):
    '''
    accuracy
    confusion metrix
    kappa
    precision
    recall
    F-score
    '''
    # 假设真实标签和预测标签
    ground_truth = y_true  # 真实标签
    predictions = y_pred  # 预测标签
    # 将列表转换为numpy格式，
    ground_truth = np.array(ground_truth, dtype=np.int32)
    predictions = np.array(predictions, dtype=np.int32)
    # 计算tp/tn/fp/fn
    tp = np.sum(np.logical_and(predictions == 1, ground_truth == 1))
    tn = np.sum(np.logical_and(predictions == 0, ground_truth == 0))
    fp = np.sum(np.logical_and(predictions == 1, ground_truth == 0))
    fn = np.sum(np.logical_and(predictions == 0, ground_truth == 1))

    # 计算accuracy
    accuracy = np.mean(predictions == ground_truth)
    # 计算recall
    recall = tp / (tp + fn)
    # 计算precision
    precision = tp / (tp + fp)
    # 计算f1-score
    f1_score = 2 * recall * precision / (recall + precision)
    # kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    con_matrix = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return accuracy, precision, recall, f1_score, kappa, con_matrix


# def logger_txt(log_dir, mode, epoch, y_true, y_pred):
#     with open(log_dir, 'a') as f:
#         accuracy, precision, recall, f1_score, kappa, con_matrix = indicators(y_true, y_pred)
#         if mode == 'val':
#             f.write('='*15 + '+'*5 + "  VAL EPOCH {} ".format(epoch) + '+'*5 + '='*15 + '\n\n')
#         if mode == 'test':
#             f.write('='*15 + '+'*5 + "  TEST EPOCH {} ".format(epoch) + '+'*5 + '='*15 + '\n\n')
#         f.write("accuracy = {}\n".format(accuracy))
#         f.write("precision = {}\n".format(precision))
#         f.write("recall = {}\n".format(recall))
#         f.write("f1_score = {}\n".format(f1_score))
#         f.write("kappa = {}\n".format(kappa))
#         f.write("con_matrix:\n")
#         f.write("{} {}\n".format(con_matrix[0][0], con_matrix[0][1]))
#         f.write("{} {}\n".format(con_matrix[1][0], con_matrix[1][1]))
#         f.write('='*15 + '+'*15 + '='*15 + '\n\n')

def logger_iteration(log_dir, iteration, f1_score):
    with open(log_dir, 'a') as f:
        f.write("iteration:{} f1_score:{}\n".format(iteration, f1_score))


def loading_data(mode, config):
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "test": transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    if mode == "train":
        image_path = config.dataset_dir

        train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                             transform=data_transform["train"])
        validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                                transform=data_transform["val"])

        # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
        flower_list = train_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in flower_list.items())
        # write dict into json file
        json_str = json.dumps(cla_dict, indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)

        nw = min([os.cpu_count(), config.batch_size if config.batch_size > 1 else 0, 8])
        print('Using {} dataloader workers every process'.format(nw))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config.batch_size, shuffle=True,
                                                   num_workers=nw)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                      batch_size=config.batch_size, shuffle=False,
                                                      num_workers=nw)

        train_num = len(train_dataset)
        val_num = len(validate_dataset)
        print("using {} images for training, {} images for validation.".format(train_num,
                                                                               val_num))

        return train_loader, validate_loader, train_num, val_num
    elif mode == "test":
        image_path = config.dataset_dir

        test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform["test"])

        nw = min([os.cpu_count(), config.batch_size if config.batch_size > 1 else 0, 8])
        print('Using {} dataloader workers every process'.format(nw))

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=config.batch_size, shuffle=True,
                                                  num_workers=nw)

        test_num = len(test_dataset)
        print("using {} images for test.".format(test_num))

        return test_loader, test_num
