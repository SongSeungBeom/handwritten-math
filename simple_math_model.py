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

trans = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,0.5, 0.5),(0.5,0.5,0.5))]) 
#45x45 크기의 이미지 자료들의 크기를 32x32로 줄여서 방대한 자료의 크기를 줄임. 이미지 사진들은 손글씨 사진들이지만 rgb이기 때문에 채널 수가 3. 0.5로 평균화했다.

trainset = torchvision.datasets.ImageFolder(root = "./loaderdata/train", transform=trans)
testset =  torchvision.datasets.ImageFolder(root = "./loaderdata/test", transform=trans)

trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=4)

classes = trainset.classes

#자료들은 자신의 클래스명으로 된 폴더에 각각 저장되어 있기 때문에 dataloader를 이용해 클래스를 저장해준다.

class Net(nn.Module): #간단한 cnn모델
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 49)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000)) #2000개씩 학습할 때마다 과정 출력
                running_loss = 0.0

    print('Finished Training')

    PATH = './model/math_net_02.pth'
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

    print('Accuracy of the network on the 180000 test images: %d %%' % (100 * correct / total))