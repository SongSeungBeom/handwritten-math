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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#gpu 가속을 위한 준비.

trans = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(227), transforms.ToTensor(), transforms.Normalize((0.5,0.5, 0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.ImageFolder(root = "./loaderdata/train", transform=trans)
testset =  torchvision.datasets.ImageFolder(root = "./loaderdata/test", transform=trans)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=64, shuffle=True, num_workers=4)

classes = trainset.classes


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, padding=0, stride=4),  # 227 -> 55
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 55 -> 27

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2, stride=1),  # 27 -> 27
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27 -> 13

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13 -> 6
        )

        self.fclayer = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 55),
        )

    def forward(self, x: torch.Tensor):
        x = self.convnet(x)
        x = torch.flatten(x, 1)
        x = self.fclayer(x)
        return x

if __name__ == '__main__':
    net = AlexNet()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(15): #15번 반복 학습

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            #gradient 매개변수를 0으로 초기화

            outputs = net(inputs)
            loss = criterion(outputs, labels) #순전파
            loss.backward() #역전파
            optimizer.step() #최적화

            running_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000)) #1000개씩 학습할 때마다 과정 출력
                running_loss = 0.0

    print('Finished Training')

    PATH = './model/55_classes_Alexnet_01.pth'
    torch.save(net.state_dict(), PATH) #모델 저장

    correct = 0
    total = 0

    with torch.no_grad(): #정확도 계산을 위한 테스트.
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() #예측과 정답이 일치하는 경우를 세서 정확도를 구한다.

    print('Accuracy of the network on the 180000 test images: %5f %%' % (100 * correct / total))