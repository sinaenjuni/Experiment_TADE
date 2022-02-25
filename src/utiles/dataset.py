from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from .data import getSubDataset
import numpy as np
import os
from PIL import Image


class CIFAR10:
    def __init__(self):
        self.classes = {'plane': 0,
                        'car': 1,
                        'bird': 2,
                        'cat': 3,
                        'deer': 4,
                        'dog': 5,
                        'frog': 6,
                        'horse': 7,
                        'ship': 8,
                        'truck': 9}
        # Image processing
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
                                             std=(0.5, 0.5, 0.5))])

        # Dataset define
        self.train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                           train=True,
                                           download=True)
        self.test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                           train=False,
                                           download=True)

        self.labels = np.array(list(self.train_dataset.targets))
        print(self.classes)
        print(self.labels)

    def getTrainDataset(self, transforms=None):
        if transforms is not None:
            self.train_dataset.transform=transforms
        else:
            self.train_dataset.transform=self.transform
        return self.train_dataset

    def getTestDataset(self, transforms=None):
        if transforms is not None:
            self.test_dataset.transform=transforms
        else:
            self.test_dataset.transform=self.transform
        return self.test_dataset

    def getTransformedDataset(self, ratio):
        return getSubDataset(dataset=self.train_dataset,
                                                   class_index=self.classes,
                                                   labels=self.labels,
                                                   lratio=ratio)


class MNIST:
    def __init__(self, image_size=32):
        self.classes = {'0': 0,
                        '1': 1,
                        '2': 2,
                        '3': 3,
                        '4': 4,
                        '5': 5,
                        '6': 6,
                        '7': 7,
                        '8': 8,
                        '9': 9}

        self.transform = transforms.Compose([transforms.Resize(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5],  # 1 for greyscale channels
                                                             std=[0.5])])
        # MNIST dataset
        self.train_dataset = torchvision.datasets.MNIST(root = '../../data/',
                                                   train = True,
                                                   download = True)
        # MNIST dataset
        self.test_dataset = torchvision.datasets.MNIST(root = '../../data/',
                                                   train = False,
                                                   download = True)
        self.labels = np.array(list(self.train_dataset.train_labels))
        print(self.classes)
        print(self.labels)

    def getTrainDataset(self, transforms=None):
        if transforms is not None:
            self.train_dataset.transform = transforms
        else:
            self.train_dataset.transform = self.transform
        return self.train_dataset

    def getTestDataset(self, transforms=None):
        if transforms is not None:
            self.test_dataset.transform = transforms
        else:
            self.test_dataset.transform = self.transform
        return self.test_dataset

    def getTransformedDataset(self, ratio):
        return getSubDataset(dataset = self.train_dataset,
                                                   class_index = self.classes,
                                                   labels = self.labels,
                                                   lratio = ratio)

# if __name__ == '__main__':
#     mnist = MNIST()
#     train_dataset = mnist.getTrainDataset()
#     train_dataset, count = mnist.getTransformedDataset(
#         [0.5 ** i for i in range(len(mnist.classes))])
#     test_dataset = mnist.getTestDataset()
#
#     print(count)
#     print(train_dataset)
#     print(test_dataset)
#
#     cifar10 = CIFAR10()
#     train_dataset = cifar10.getTrainDataset()
#     transforms_dataset, count = cifar10.getTransformedDataset(
#         [0.5 ** i for i in range(len(cifar10.classes))])
#     test_dataset = cifar10.getTestDataset()
#
#     print(count)
#     print(train_dataset)
#     print(test_dataset)


class ImageNetLT(Dataset):
    def __init__(self, type='train', transform=None):
        BASEPATH = '../../data/ImageNet_LT'
        self.img_path = os.path.join(BASEPATH, 'ImageNet_LT_open')
        print(os.listdir(self.img_path))
        if type == 'train': data_path = os.path.join(BASEPATH, 'ImageNet_LT_train.txt')
        elif type == 'val': data_path = os.path.join(BASEPATH, 'ImageNet_LT_val.txt')
        elif type == 'test': data_path = os.path.join(BASEPATH, 'ImageNet_LT_test.txt')
        else: assert False, 'type should be selected in train, val, test'


        with open(data_path, 'r') as f:
            data_file = f.readlines()

        self.data_file = list(map(lambda x: ['ILSVRC2010_val_'+
                                        x.split(' ')[0]
                                        .split('_')[1]
                                        .split('.')[0]
                                        .zfill(8)+'.JPEG',
                                        int(x.split(' ')[1])], data_file))
            # data_file = list(map(lambda x: [x[0].split('_')[1], x[1]], data_file))

        self.class_list = [i[1] for i in self.data_file]
        self.classes = list(set(self.class_list))
        self.class_count = {classes:self.class_list.count(classes) for classes in self.classes}

        # print(self.img_path)
        # print(self.data_file[100])
        # print(self.class_list)
        # print(self.classes)
        # print(self.class_count)

        self.transform = transform

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, index):
        img, label = self.data_file[index]
        img = os.path.join(self.img_path, img)
        print(img)
        img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label





# if __name__ == '__main__':
#     imageNet_LT = ImageNetLT()
#
#     img, label = imageNet_LT[1]
#     print(img, label)