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
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


trans = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(227), transforms.ToTensor(), transforms.Normalize((0.5,0.5, 0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.ImageFolder(root = "./loaderdata/train", transform=trans)
testset1 = torchvision.datasets.ImageFolder(root = "./testing_img_extracting/98+76", transform=trans)
testset2 = torchvision.datasets.ImageFolder(root = "./testing_img_extracting/23+cos(75)", transform=trans)
testset3 = torchvision.datasets.ImageFolder(root = "./testing_img_extracting/integral x+3 dx", transform=trans)


trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)
testloader1 = DataLoader(testset1, batch_size=16, shuffle= False, num_workers=4)
testloader2 = DataLoader(testset2, batch_size=16, shuffle= False, num_workers=4)
testloader3 = DataLoader(testset3, batch_size=16, shuffle= False, num_workers=4)

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

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    net = AlexNet()
    net.load_state_dict(torch.load('../model/55_classes_Alexnet_01.pth'))

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