import os
import torch
import torch.nn as nn
from itertools import product

DATA_TYPE   = f'cifar10/aug/p10/'
GAN_PATH    = f'../../weights/WGAN-GP/{DATA_TYPE}'
REFER_PATH  = f'../../weights/reference/'
SAVE_PATH   = f'../../weights/our_weights/{DATA_TYPE}'


file_list = os.listdir(GAN_PATH)
print(f"Find the number of {len(file_list)} GAN weight files")


GAN_PATH    = os.path.join(GAN_PATH, "D_10.pth")
GAN_FILE    = torch.load(GAN_PATH)

REFER_PATH = os.path.join(REFER_PATH, "TADE_resnet32.pth")
REFER_FILE = torch.load(REFER_PATH)
MODI_FILE  = torch.load(REFER_PATH)


print(len(GAN_FILE))
print(len(REFER_FILE))



target=[]
target_layer = ["layer1", "layer2", "layer3"]
target_expert = ["s.0", "s.1", "s.2"]

print(list(product(target_layer, 1)))

# print(combinations(target, target_expert))
# print(list(product(*[target_layer, target_expert])))

assert False
count = 0

for i, (k, v) in enumerate(GAN_FILE.items()):
    if any(target in k for target in target_layer):
        if "layer1" in k:
            print("layer1")
            print(k, type(MODI_FILE[k]))
            MODI_FILE[k] = v
            count+=1
        else:
            print("Not layer1")
            for expert in target_expert:
                key = k.split('.')
                key[0] = f"{key[0]}{expert}"
                key = '.'.join(key)
                print(key, type(MODI_FILE[key]))
                MODI_FILE[k] = v
                count+=1

    if ("layer" not in k) and ("last" not in k) and ("linear" not in k):
        print("Not layer")
        print(k, type(MODI_FILE[k]))
        MODI_FILE[k] = v
        count+=1

print(count)