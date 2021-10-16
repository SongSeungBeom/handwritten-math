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
import time
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0")
#gpu 가속을 위한 준비.

trans = transforms.Compose([transforms.Resize(227), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.5,0.5, 0.5),(0.5,0.5,0.5))])
#trans = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

trainset = torchvision.datasets.ImageFolder(root = "./loaderdata/train", transform=trans)
testset =  torchvision.datasets.ImageFolder(root = "./loaderdata/test", transform=trans)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=True)

classes = trainset.classes

class conv_bn_relu(nn.Module):
    def __init__(self, nin, nout, kernel_size, stride, padding, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)

        return out


class Transition_layer(nn.Sequential):
    def __init__(self, nin, theta=1):
        super(Transition_layer, self).__init__()

        self.add_module('conv_1x1',
                        conv_bn_relu(nin=nin, nout=int(nin * theta), kernel_size=1, stride=1, padding=0, bias=False))
        self.add_module('avg_pool_2x2', nn.AvgPool2d(kernel_size=2, stride=2, padding=0))


class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()

        self.conv_3x3_first = conv_bn_relu(nin=3, nout=32, kernel_size=3, stride=2, padding=1, bias=False)

        self.conv_1x1_left = conv_bn_relu(nin=32, nout=16, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_3x3_left = conv_bn_relu(nin=16, nout=32, kernel_size=3, stride=2, padding=1, bias=False)

        self.max_pool_right = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv_1x1_last = conv_bn_relu(nin=64, nout=32, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out_first = self.conv_3x3_first(x)

        out_left = self.conv_1x1_left(out_first)
        out_left = self.conv_3x3_left(out_left)

        out_right = self.max_pool_right(out_first)

        out_middle = torch.cat((out_left, out_right), 1)

        out_last = self.conv_1x1_last(out_middle)

        return out_last


class dense_layer(nn.Module):
    def __init__(self, nin, growth_rate, drop_rate=0.2):
        super(dense_layer, self).__init__()

        self.dense_left_way = nn.Sequential()

        self.dense_left_way.add_module('conv_1x1',
                                       conv_bn_relu(nin=nin, nout=growth_rate * 2, kernel_size=1, stride=1, padding=0,
                                                    bias=False))
        self.dense_left_way.add_module('conv_3x3',
                                       conv_bn_relu(nin=growth_rate * 2, nout=growth_rate // 2, kernel_size=3, stride=1,
                                                    padding=1, bias=False))

        self.dense_right_way = nn.Sequential()

        self.dense_right_way.add_module('conv_1x1',
                                        conv_bn_relu(nin=nin, nout=growth_rate * 2, kernel_size=1, stride=1, padding=0,
                                                     bias=False))
        self.dense_right_way.add_module('conv_3x3_1',
                                        conv_bn_relu(nin=growth_rate * 2, nout=growth_rate // 2, kernel_size=3,
                                                     stride=1, padding=1, bias=False))
        self.dense_right_way.add_module('conv_3x3 2',
                                        conv_bn_relu(nin=growth_rate // 2, nout=growth_rate // 2, kernel_size=3,
                                                     stride=1, padding=1, bias=False))

        self.drop_rate = drop_rate

    def forward(self, x):
        left_output = self.dense_left_way(x)
        right_output = self.dense_right_way(x)

        if self.drop_rate > 0:
            left_output = F.dropout(left_output, p=self.drop_rate, training=self.training)
            right_output = F.dropout(right_output, p=self.drop_rate, training=self.training)

        dense_layer_output = torch.cat((x, left_output, right_output), 1)

        return dense_layer_output


class DenseBlock(nn.Sequential):
    def __init__(self, nin, num_dense_layers, growth_rate, drop_rate=0.0):
        super(DenseBlock, self).__init__()

        for i in range(num_dense_layers):
            nin_dense_layer = nin + growth_rate * i
            self.add_module('dense_layer_%d' % i,
                            dense_layer(nin=nin_dense_layer, growth_rate=growth_rate, drop_rate=drop_rate))


class PeleeNet(nn.Module):
    def __init__(self, growth_rate=32, num_dense_layers=[3, 4, 8, 6], theta=1, drop_rate=0.0, num_classes=55):
        super(PeleeNet, self).__init__()

        assert len(num_dense_layers) == 4

        self.features = nn.Sequential()
        self.features.add_module('StemBlock', StemBlock())

        nin_transition_layer = 32

        for i in range(len(num_dense_layers)):
            self.features.add_module('DenseBlock_%d' % (i + 1),
                                     DenseBlock(nin=nin_transition_layer, num_dense_layers=num_dense_layers[i],
                                                growth_rate=growth_rate, drop_rate=0.0))
            nin_transition_layer += num_dense_layers[i] * growth_rate

            if i == len(num_dense_layers) - 1:
                self.features.add_module('Transition_layer_%d' % (i + 1),
                                         conv_bn_relu(nin=nin_transition_layer, nout=int(nin_transition_layer * theta),
                                                      kernel_size=1, stride=1, padding=0, bias=False))
            else:
                self.features.add_module('Transition_layer_%d' % (i + 1),
                                         Transition_layer(nin=nin_transition_layer, theta=1))

        self.linear = nn.Linear(nin_transition_layer, num_classes)

    def forward(self, x):
        stage_output = self.features(x)

        global_avg_pool_output = F.adaptive_avg_pool2d(stage_output, (1, 1))
        global_avg_pool_output_flat = global_avg_pool_output.view(global_avg_pool_output.size(0), -1)

        output = self.linear(global_avg_pool_output_flat)

        return output

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


# calculate the metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


# calculate the loss per mini-batch
def loss_batch(loss_func, output, target, opt=None):
    loss_b = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()

    return loss_b.item(), metric_b


# calculate the loss per epochs
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


# function to start training
def train_val(model, params):
    num_epochs = params['num_epochs']
    loss_func = params['loss_func']
    opt = params['optimizer']
    train_dl = params['train_dl']
    val_dl = params['val_dl']
    sanity_check = params['sanity_check']
    lr_scheduler = params['lr_scheduler']
    path2weights = params['path2weights']

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr= {}'.format(epoch, num_epochs - 1, current_lr))

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print('Copied best model weights!')

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print('Loading best model weights!')
            model.load_state_dict(best_model_wts)

        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' % (
        train_loss, val_loss, 100 * val_metric, (time.time() - start_time) / 60))
        print('-' * 10)

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history



if __name__ == '__main__':
    model = PeleeNet()
    model.to(device)

    loss_func = nn.CrossEntropyLoss(reduction='sum')
    opt = optim.Adam(model.parameters(), lr=0.01)

    lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)
    params_train = {
        'num_epochs': 10,
        'optimizer': opt,
        'loss_func': loss_func,
        'train_dl': trainloader,
        'val_dl': testloader,
        'sanity_check': False,
        'lr_scheduler': lr_scheduler,
        'path2weights': './model/peleeNet.pt',
    }

    model, loss_hist, metric_hist = train_val(model, params_train)