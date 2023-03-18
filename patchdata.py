import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch

class PatchGenerator:

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, img):
        num_channels = img.size(0) #이미지 사이즈의 첫번째가 채널 수
        #unfold함수를 이용하여 dimension 1방향,2방향으로 순차적으로 자름
        #3x16x16x16x16
        patches = img.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size).reshape(num_channels, -1, self.patch_size, self.patch_size) #256이 들어가도록 설정
        patches = patches.permute(1,0,2,3) #256x3x16x16
        num_patch = patches.size(0)

        return patches.reshape(num_patch,-1) #각 patch를 1열로 피는 작업

class Flattened2Dpatches:

    def __init__(self, patch_size=16, dataname='imagenet', img_size=256, batch_size=64): #img_size는 resize할 이미지
        self.patch_size = patch_size
        self.dataname = dataname #데이터에 따라 불러오는 방법을 다르게
        self.img_size = img_size #만약 입력값의 이미지 가로 세로가 다르면 (256,256)이런식으로 설정
        self.batch_size = batch_size

    def make_weights(self, labels, nclasses):
        labels = np.array(labels)
        weight_arr = np.zeros_like(labels)
        _, counts = np.unique(labels, return_counts=True)
        for cls in range(nclasses):
            weight_arr = np.where(labels == cls, 1/counts[cls], weight_arr) 
    
        return weight_arr 

    def patchdata(self):
        #cifar 10에 대한 평균과 표준편차
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        train_transform = transforms.Compose([transforms.Resize(self.img_size), transforms.RandomCrop(self.img_size, padding=2),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std),
                                              PatchGenerator(self.patch_size)]) #위에서 따로 설정
        #text는 data augmentation할 필요x
        test_transform = transforms.Compose([transforms.Resize(self.img_size), transforms.ToTensor(),
                                             transforms.Normalize(mean, std), PatchGenerator(self.patch_size)])

        if self.dataname == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
            #testset의 반은 test 반은 validation
            evens = list(range(0, len(testset), 2))
            odds = list(range(1, len(testset), 2))
            valset = torch.utils.data.Subset(testset, evens) #짝수 index
            testset = torch.utils.data.Subset(testset, odds) #홀수 index
          
        elif self.dataname == 'imagenet':
            pass

        #클래스를 뽑을 확률을 동일하게
        weights = self.make_weights(trainset.targets, len(trainset.classes))  # 가중치 계산
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)) 
        trainloader = DataLoader(trainset, batch_size=self.batch_size, sampler=sampler)
        valloader = DataLoader(valset, batch_size=self.batch_size, shuffle=False)
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)

        return trainloader, valloader, testloader

def imshow(img):
    plt.figure(figsize=(100,100))
    plt.imshow(img.permute(1,2,0).numpy())
    plt.savefig('pacth_example.png')

if __name__ == "__main__":
    print("Testing Flattened2Dpatches..")
    batch_size = 64
    patch_size = 8
    img_size = 32
    num_patches = int((img_size*img_size)/(patch_size*patch_size))
    #테스트시 평균과 표준편차 지워서 했음
    d = Flattened2Dpatches(dataname='cifar10', img_size=img_size, patch_size=patch_size, batch_size=batch_size)
    trainloader, _, _ = d.patchdata()
    images, labels = iter(trainloader).next()
    print(images.size(), labels.size())

    sample = images.reshape(batch_size, num_patches, -1, patch_size, patch_size)[0]
    print("Sample image size: ", sample.size())
    imshow(torchvision.utils.make_grid(sample, nrow=int(img_size/patch_size)))
    plt.show()