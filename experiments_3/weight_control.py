import os
import torch
import torch.nn as nn
name = 'experiments3/resnet_tade/'

resnet_s_PATH = f'weights/experiments2/Resnet_s/GAN/D_10.pth'
resnet_s_weight = torch.load(resnet_s_PATH)

resnet_tade_PATH = "/home/sin/git/pytorch.GAN/weights/experiments3/Resnet_tade/classifier/model.pth"
resnet_tade_weight = torch.load(resnet_tade_PATH)

modified_tade_weight = torch.load(resnet_tade_PATH)

# model.load_state_dict(torch.load(SAVE_PATH), strict=False)
if __name__ == "__main__":
    # count = 0
    print(resnet_s_weight)
    print(len(resnet_s_weight))
    print(len(resnet_tade_weight))

    tade_layer = ["layer2", "layer3"]
    t = "layer"
    # print(any(target in t for target in tade_layer))

    for k, v in resnet_s_weight.items():
        if not any(target in k for target in tade_layer):
            modified_tade_weight[k] = v
        else:
            for i in range(0, 3, 2):
                keyword = k.split(".")
                keyword[0] = keyword[0] + "s"
                keyword[1] = f"{i}." + keyword[1]
                tade_key = '.'.join(keyword)
                modified_tade_weight[tade_key] = v
        # try:
        #     # print(resnet_tade_weight[k].size() == v.size())
        #     resnet_tade_weight[k] = v
        #     # print(k)
        #     # print(resnet_tade_weight[k].size())
        # except:
        #     # if "layer" in k:
        #     for i in range(3):
        #         keyword = k.split(".")
        #         keyword[0] = keyword[0] + "s"
        #         keyword[1] = f"{i}." + keyword[1]
        #         tade_key = '.'.join(keyword)
        #
        #         # resnet_tade_weight[tade_key] = v
        #         # print(v.size())
        #         # print(resnet_tade_weight[tade_key].size())
        #         print(v.size() == resnet_tade_weight[tade_key].size())
            # pass

    # print("len resent_s", len(resnet_s_weight))
    # print("len resnet_tade", len(resnet_tade_weight))
    # print("len resent_tade modified", len(modified_tade_weight))
    #
    #
    # print(resnet_s_weight.keys())
    # ret = []
    # for k in modified_tade_weight.keys():
    #     if k not in resnet_tade_weight.keys():
    #         print(k)

    # Save modified weight
    SAVE_PATH = f'weights/{name}/'
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    torch.save(resnet_tade_weight, SAVE_PATH + f'weight_control.pth')



    # print(resnet_s_weight["conv1.weight"] == resnet_tade_weight["conv1.weight"])
    # print(resnet_tade_weight["layer2s.0.0.conv1.weight"] == resnet_tade_weight["layer2s.1.0.conv1.weight"] )
    # print()