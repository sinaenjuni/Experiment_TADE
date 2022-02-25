
import torch
import random
import numpy as np
import os, sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from PIL import Image
from .imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100


class BalancedSampler(Sampler):
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size

    def __iter__(self):
        count = self.__len__()
        print('sampler', count)
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets])  # AcruQRally we need to upscale to next full batch
        else:
            return max([len(bucket) for bucket in
                        self.buckets]) * self.bucket_num  # Ensures every instance has the chance to be visited in an epoch


class ImbalanceCIFAR10DataLoader(DataLoader):
    """
    Imbalance Cifar10 Data Loader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True, balanced=False,
                 retain_epoch_size=True, imb_factor=0.01):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
        print(train_trsfm)
        test_trsfm = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        if training:
            dataset = IMBALANCECIFAR10(data_dir, train=True, download=True, transform=train_trsfm, imb_factor=imb_factor)
            val_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_trsfm)  # test set
        else:
            dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_trsfm)  # test set
            val_dataset = None

        self.dataset = dataset
        self.val_dataset = val_dataset

        num_classes = len(np.unique(dataset.targets))
        assert num_classes == 10

        cls_num_list = [0] * num_classes
        for label in dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list

        if balanced:
            if training:
                buckets = [[] for _ in range(num_classes)]
                for idx, label in enumerate(dataset.targets):
                    buckets[label].append(idx)
                sampler = BalancedSampler(buckets, retain_epoch_size)
                shuffle = False
            else:
                print("Test set will not be evaluated with balanced sampler, nothing is done to make it balanced")
        else:
            sampler = None

        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs,
                         sampler=sampler)  # Note that sampler does not apply to validation set

    def split_validation(self):
        # If you do not want to validate:
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    loader = ImbalanceCIFAR10DataLoader(data_dir='../../data', batch_size=64,
                                        shuffle=True, num_workers=4, training=True,
                                        imb_factor=0.01)
    print(len(loader))
    for idx, data, in enumerate(loader):
        img, label = data
        if idx == 1:
            break
        else:
            grid = make_grid(img)
            grid = (grid + 1) / 2
            grid.clamp(0, 1)
            print(grid.size())
            plt.imshow(grid.permute(1,2,0))
            plt.show()
            print(idx, data[1])
