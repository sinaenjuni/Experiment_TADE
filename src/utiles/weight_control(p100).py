import os
import torch
import torch.nn as nn
from itertools import product, combinations, chain


TARGET_EPOCH = 100

DATA_TYPE   = f'cifar10/aug/p100/'
GAN_PATH    = f'../../weights/WGAN-GP/{DATA_TYPE}'
REFER_PATH  = f'../../weights/reference/'
SAVE_PATH   = f'../../weights/our_weights/{DATA_TYPE}'


file_list = os.listdir(GAN_PATH)
print(f"Find the number of {len(file_list)} GAN weight files")


GAN_PATH    = os.path.join(GAN_PATH, f"D_{TARGET_EPOCH}.pth")
GAN_FILE    = torch.load(GAN_PATH)

REFER_PATH = os.path.join(REFER_PATH, "TADE_resnet32.pth")
REFER_FILE = torch.load(REFER_PATH)
# MODI_FILE  = torch.load(REFER_PATH)


print(len(GAN_FILE))
print(len(REFER_FILE))


target_layer = ["layer1", "layer2", "layer3"]
target_expert = ["s.0", "s.1", "s.2"]

target_layer = sum([list(combinations(target_layer, i+1)) for i in range(len(target_layer))], [])
target_expert = sum([list(combinations(target_expert, i+1)) for i in range(len(target_expert))], [])

for i in product(target_layer, target_expert):
    print(i)

# for i in product(target_layer, target_expert):
#     MODI_FILE = torch.load(REFER_PATH)
#     # print(i[0], i[1])
#     target_layer_ = i[0]
#     target_expert_ = i[1]
#
#     count = 0
#     for i, (k, v) in enumerate(GAN_FILE.items()):
#         if any(target in k for target in target_layer_):
#             if "layer1" in k:
#                 # print("layer1")
#                 # print(k, type(MODI_FILE[k]))
#                 MODI_FILE[k] = v
#                 count+=1
#             else:
#                 # print("Not layer1")
#                 for expert in target_expert_:
#                     key = k.split('.')
#                     key[0] = f"{key[0]}{expert}"
#                     key = '.'.join(key)
#                     # print(key, type(MODI_FILE[key]))
#                     MODI_FILE[key] = v
#                     count+=1
#
#         if ("layer" not in k) and ("last" not in k) and ("linear" not in k):
#             # print("Not layer")
#             # print(k, type(MODI_FILE[k]))
#             MODI_FILE[k] = v
#             count+=1
#
#     print("Count", count)
#     print("Len of all parame", len(MODI_FILE))
    # if not os.path.exists(SAVE_PATH):
    #     os.makedirs(SAVE_PATH)
    # torch.save(MODI_FILE, SAVE_PATH + f"{TARGET_EPOCH}_{target_layer_}_{target_expert_}.pth")