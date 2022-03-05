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
from models.expert_resnet_cifar import resnet32
from utiles.loss import DiverseExpertLoss


from utiles.metric import ClassificationMetric
from utiles.pretty_confusion_matrix import pp_matrix

from itertools import product, combinations


# Define hyper-parameters
name = 'pre-training/TADE/cifar10/aug/p10/'
tensorboard_path = f'../../tb_logs/{name}'

# Argument with loading pre-training weights
TARGET_EPOCH = 100
DATA_TYPE    = f'cifar10/aug/p10/'
WEIGHTS_PATH = f'../../weights/our_weights/{DATA_TYPE}'

target_layer = ["layer1", "layer2", "layer3"]
target_expert = ["s.0", "s.1", "s.2"]

target_layer = sum([list(combinations(target_layer, i+1)) for i in range(len(target_layer))], [])
target_expert = sum([list(combinations(target_expert, i+1)) for i in range(len(target_expert))], [])



# Argument with model training
num_workers = 4
num_epochs = 400
batch_size = 128
imb_factor = 0.1
num_class = 10
learning_rate = 0.1
weight_decay = 5e-4
momentum = 0.9
nesterov = True

return_feature = True


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# Define Tensorboard
# tb = getTensorboard(tensorboard_path)


for i in product(target_layer, target_expert):
    target_layer_ = i[0]
    target_expert_ = i[1]
    pre_trained_weight = torch.load(WEIGHTS_PATH + f"{TARGET_EPOCH}_{target_layer_}_{target_expert_}.pth")
    print(len(pre_trained_weight))
    tb = getTensorboard(tensorboard_path + f"{TARGET_EPOCH}_{target_layer_}_{target_expert_}/")

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

    train_metric = ClassificationMetric([i for i in range(10)])
    test_metric = ClassificationMetric([i for i in range(10)])


    # Define model
    model = resnet32(num_classes=10, use_norm=True).to(device)
    print(model)
    # SAVE_PATH = f'../../weights/{name}/'
    # if not os.path.exists(SAVE_PATH):
    #     os.makedirs(SAVE_PATH)
    # torch.save(model.state_dict(), SAVE_PATH + f'model.pth')

    criterion = DiverseExpertLoss(cls_num_list=cls_num_list, tau=4).to(device)

    # SAVE_PATH = f'../../weights/experiments3/resnet_tade/weight_control.pth'
    model.load_state_dict(pre_trained_weight, strict=True)




    # Define optimizer
    # optimizer = torch.optim.Adam(model.parameters(),
    #                             lr=learning_rate,
    #                             # momentum=momentum,
    #                             weight_decay=weight_decay)

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

        for train_idx, data in enumerate(train_data_loader):
            img, target = data
            img, target = img.to(device), target.to(device)
            batch = img.size(0)

            optimizer.zero_grad()

            model.train()
            extra_info = {}
            output = model(img)

            logits = output["logits"]
            extra_info.update({
                "logits": logits.transpose(0, 1)
            })

            output = output["output"]
            loss = criterion(output_logits=output, target=target, extra_info=extra_info)
            loss.backward()
            optimizer.step()


            train_loss = np.append(train_loss, loss.item())
            pred = output.argmax(dim=1)

            train_true = np.append(train_true, target.cpu().numpy())
            train_pred = np.append(train_pred, pred.cpu().numpy())

        test_true = np.array([])
        test_pred = np.array([])
        model.eval()
        with torch.no_grad():
            for test_idx, data in enumerate(test_data_loader):
                img, target = data
                img, target = img.to(device), target.to(device)
                batch = img.size(0)

                output = model(img)
                logits = output["logits"]
                extra_info.update({
                    "logits": logits.transpose(0, 1)
                })
                output = output["output"]

                loss = criterion(output_logits=output, target=target, extra_info=extra_info)
                # loss = F.cross_entropy(pred, target)
                test_loss = np.append(test_loss, loss.item())

                pred = output.argmax(-1)

                test_true = np.append(test_true, target.cpu().numpy())
                test_pred = np.append(test_pred, pred.cpu().numpy())

        test_result = test_metric.calcMetric(epoch + 1, test_true, test_pred)
        tb.add_text(tag='log', global_step=epoch + 1, text_string=test_result['text'])
        fig = pp_matrix(pd.DataFrame(test_result['best_cm']), figsize=(11, 11))
        tb.add_figure(tag="best_cm", figure=fig, global_step=epoch+1)

        print(max([param_group['lr'] for param_group in optimizer.param_groups]),
                    min([param_group['lr'] for param_group in optimizer.param_groups]))
        lr_scheduler.step()







