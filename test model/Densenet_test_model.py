from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import copy
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


trans = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5,0.5, 0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.ImageFolder(root = "./loaderdata/train", transform=trans)
testset1 = torchvision.datasets.ImageFolder(root = "./testing_img_extracting/98+76", transform=trans)
testset2 = torchvision.datasets.ImageFolder(root = "./testing_img_extracting/23+cos(75)", transform=trans)
testset3 = torchvision.datasets.ImageFolder(root = "./testing_img_extracting/integral x+3 dx", transform=trans)


trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)
testloader1 = DataLoader(testset1, batch_size=16, shuffle= False, num_workers=4)
testloader2 = DataLoader(testset2, batch_size=16, shuffle= False, num_workers=4)
testloader3 = DataLoader(testset3, batch_size=16, shuffle= False, num_workers=4)

classes = trainset.classes

from torch.optim.lr_scheduler import ReduceLROnPlateau

class BottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inner_channels = 4 * growth_rate

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, inner_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, growth_rate, 3, stride=1, padding=1, bias=False)
        )

        self.shortcut = nn.Sequential()

    def forward(self, x):
        return torch.cat([self.shortcut(x), self.residual(x)], 1)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)


class DenseNet(nn.Module):
    def __init__(self, nblocks, growth_rate=12, reduction=0.5, num_classes=55, init_weights=True):
        super().__init__()

        self.growth_rate = growth_rate
        inner_channels = 2 * growth_rate  # output channels of conv1 before entering Dense Block

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, inner_channels, 7, stride=2, padding=3),
            nn.MaxPool2d(3, 2, padding=1)
        )

        self.features = nn.Sequential()

        for i in range(len(nblocks) - 1):
            self.features.add_module('dense_block_{}'.format(i), self._make_dense_block(nblocks[i], inner_channels))
            inner_channels += growth_rate * nblocks[i]
            out_channels = int(reduction * inner_channels)
            self.features.add_module('transition_layer_{}'.format(i), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module('dense_block_{}'.format(len(nblocks) - 1),
                                 self._make_dense_block(nblocks[len(nblocks) - 1], inner_channels))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU())

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(inner_channels, num_classes)

        # weight initialization
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def _make_dense_block(self, nblock, inner_channels):
        dense_block = nn.Sequential()
        for i in range(nblock):
            dense_block.add_module('bottle_neck_layer_{}'.format(i), BottleNeck(inner_channels, self.growth_rate))
            inner_channels += self.growth_rate
        return dense_block

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def DenseNet_121():
    return DenseNet([6, 12, 24, 6])


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


def loss_batch(loss_func, output, target, opt=None):
    loss_b = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()

    return loss_b.item(), metric_b


def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)

        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b

        if metric_b is not None:
            running_metric += metric_b

        if sanity_check is True:
            break

    loss = running_loss / len_data
    metric = running_metric / len_data
    return loss, metric

if __name__ == '__main__':
    net = DenseNet_121()
    net.load_state_dict(torch.load('../model/55_classes_densenet_02.pt'))

    start = time.time()

    #98+76
    dataiter1 = iter(testloader1)
    images1, labels1 = dataiter1.next()

    # 이미지를 출력합니다.
    #imshow(torchvision.utils.make_grid(images1))

    outputs1 = net(images1)

    _, predicted1 = torch.max(outputs1, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted1[j]] for j in range(5)))


    #23+cos(75)
    dataiter2 = iter(testloader2)
    images2, labels2 = dataiter2.next()

    #imshow(torchvision.utils.make_grid(images2))

    outputs2 = net(images2)

    _, predicted2 = torch.max(outputs2, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted2[j]] for j in range(10)))


    # integral x+3 dx
    dataiter3 = iter(testloader3)
    images3, labels3 = dataiter3.next()

    #imshow(torchvision.utils.make_grid(images3))

    outputs3 = net(images3)

    _, predicted3 = torch.max(outputs3, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted3[j]] for j in range(6)))

    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간