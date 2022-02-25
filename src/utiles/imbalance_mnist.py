# From: https://github.com/kaidic/LDAM-DRW
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


class IMBALANCEMNIST(torchvision.datasets.MNIST):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False, reverse=False):
        super(IMBALANCEMNIST, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor, reverse)
        self.gen_imbalanced_data(img_num_list)
        self.reverse = reverse

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, reverse):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                if reverse:
                    num = img_max * (imb_factor ** (((cls_num - 1) - cls_idx) / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
                else:
                    num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            # np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            # new_targets.extend([the_class, ] * the_img_num)
            new_targets.extend(targets_np[selec_idx])

        new_data = np.vstack(new_data)
        self.data = torch.tensor(new_data)
        self.targets = torch.tensor(new_targets)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


if __name__ == '__main__':
    from PIL import Image

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])

    train_dataset = IMBALANCEMNIST(root='../../data', train=True, imb_factor=0.01,
                                   download=True, transform=transform)

    test_dataset = torchvision.datasets.MNIST(root='../../data', train=False,
                                              download=True, transform=transform)



    # print(type(test_dataset.data))
    # print(train_dataset.data.shape)
    # images = Image.fromarray(train_dataset.data[0].numpy(), mode='L')
    # print(images)
    # print(train_dataset[0])


    # import numpy as np
    # train_labels = train_dataset.train_labels
    # train_labels = np.array(train_labels)
    # for i in np.unique(train_labels):
    #     print(i, len(train_labels[train_labels == i]))

    # print(len(torchvision.datasets.MNIST(root='../../data', train=True, download=True)))

    # trainloader = iter(trainset)
    # data, label = next(trainloader)

    # import pdb;
    # pdb.set_trace()