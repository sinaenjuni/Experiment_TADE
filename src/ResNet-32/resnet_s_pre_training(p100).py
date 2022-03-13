import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
# import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
# import seaborn as sns

from utiles.tensorboard import getTensorboard
from utiles.data import getSubDataset
from utiles.imbalance_cifar10_loader import ImbalanceCIFAR10DataLoader
from models.resnet_s import resnet32

from utiles.metric import ClassificationMetric
from utiles.pretty_confusion_matrix import pp_matrix



# Define hyper-parameters
name = 'resnet_s/pre-training/cifar10/aug/p100/'
tensorboard_path = f'../../tb_logs/{name}'

TARGET_EPOCHS = ["25",
                 "50",
                 "75",
                 "150",
                 "175",
                 "200"]
WEIGHTS_PATH = "../../weights/WGAN-GP/cifar10/aug/p100/"


num_workers = 4
num_epochs = 400
batch_size = 128
imb_factor = 0.01
num_class = 10
learning_rate = 0.1
weight_decay = 5e-4
momentum = 0.9
nesterov = True



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)


# Define DataLoader
train_data_loader = ImbalanceCIFAR10DataLoader(data_dir='../../data',
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               training=True,
                                               imb_factor=imb_factor)


test_data_loader = ImbalanceCIFAR10DataLoader(data_dir='../../data',
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              training=False)


print("Number of train dataset", len(train_data_loader.dataset))
print("Number of test dataset", len(test_data_loader.dataset))

print(train_data_loader.cls_num_list)
cls_num_list = train_data_loader.cls_num_list



for target_epoch in TARGET_EPOCHS:
    # Define Tensorboard
    tb = getTensorboard(tensorboard_path + f"TE_{target_epoch}/")

    weight_file = torch.load(os.path.join(WEIGHTS_PATH, f"D_{target_epoch}.pth"))

    train_metric = ClassificationMetric([i for i in range(10)])
    test_metric = ClassificationMetric([i for i in range(10)])


    # Define model
    model = resnet32(num_classes=10, use_norm=True).to(device)
    print(model)

    # SAVE_PATH = f'../../weights/experiments2/Resnet_s/GAN/D_200.pth'
    # model.load_state_dict(torch.load(SAVE_PATH), strict=False)
    model.load_state_dict(weight_file, strict=False)

    optimizer = torch.optim.SGD(model.parameters(),
                                momentum=momentum,
                                lr=learning_rate,
                                weight_decay=weight_decay,
                                nesterov=nesterov)

    step1 = 160
    step2 = 180
    gamma = 0.1
    warmup_epoch = 5

    def lr_lambda(epoch):
        if epoch >= step2:
            lr = gamma * gamma
        elif epoch >= step1:
            lr = gamma
        else:
            lr = 1

        """Warmup"""
        if epoch < warmup_epoch:
            lr = lr * float(1 + epoch) / warmup_epoch
        print(lr)
        return lr

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



    # Training model
    for epoch in range(num_epochs):
        train_loss = np.array([])
        test_loss = np.array([])
        train_true = np.array([])
        train_pred = np.array([])
        test_true = np.array([])
        test_pred = np.array([])

        model.train()
        for train_idx, data in enumerate(train_data_loader):
            img, target = data
            img, target = img.to(device), target.to(device)
            batch = img.size(0)

            optimizer.zero_grad()

            pred = model(img)

            loss = F.cross_entropy(pred, target)
            train_loss = np.append(train_loss, loss.item())

            loss.backward()
            optimizer.step()

            pred = pred.argmax(-1)

            train_true = np.append(train_true, target.cpu().numpy())
            train_pred = np.append(train_pred, pred.cpu().numpy())

        model.eval()
        with torch.no_grad():
            for test_idx, data in enumerate(test_data_loader):
                img, target = data
                img, target = img.to(device), target.to(device)
                batch = img.size(0)

                pred = model(img)
                loss = F.cross_entropy(pred, target)
                test_loss = np.append(test_loss, loss.item())

                pred = pred.argmax(-1)

                test_true = np.append(test_true, target.cpu().numpy())
                test_pred = np.append(test_pred, pred.cpu().numpy())


        test_result = test_metric.calcMetric(epoch + 1, test_true, test_pred)
        tb.add_text(tag='log', global_step=epoch + 1, text_string=test_result['text'])

        fig = pp_matrix(pd.DataFrame(test_result['best_cm']), figsize=(11, 11))
        tb.add_figure(tag="best_cm", figure=fig, global_step=epoch+1)

        tb.flush()

        print(max([param_group['lr'] for param_group in optimizer.param_groups]),
                    min([param_group['lr'] for param_group in optimizer.param_groups]))
        lr_scheduler.step()


