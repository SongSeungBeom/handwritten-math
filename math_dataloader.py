from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms

def train():

    trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5,),(0.5,))])

    trainset = torchvision.datasets.ImageFolder(root = "./loaderdata/train", transform=trans)
    testset =  torchvision.datasets.ImageFolder(root = "./loaderdata/test", transform=trans)

    classes = trainset.classes

    trainloader = DataLoader(trainset, batch_size = 16, shuffle= True, num_workers=4)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    print(images.shape)
    imshow(torchvision.utils.make_grid(images, nrow=4))
    print(images.shape)
    print((torchvision.utils.make_grid(images)).shape)
    print("".join("%5s " % classes[labels[j]] for j in range(16)))

def imshow(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()

if __name__ =='__main__':
    train()