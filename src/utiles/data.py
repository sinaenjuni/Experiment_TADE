import numpy as np
from torch.utils.data import Subset, ConcatDataset

def getSubDataset(dataset, class_index:dict, labels, lratio:list):
    transformed_dataset = []
    count_dataset = {'class':[], 'original': [], 'transformed': []}

    for i, (name, idx) in enumerate(class_index.items()):
        target_label_indeces = np.where(labels == idx)[0]
        # print(i, target_label_indeces)
        target_subset = Subset(dataset, target_label_indeces)
        len_dataset = len(target_subset)

        ratio = len_dataset * (1 * lratio[i])
        ratio = int(ratio)
        transformed_subset = Subset(target_subset, range(ratio))
        len_transformed = len(transformed_subset)
        # print(len(transformed_subset))

        count_dataset['class'] += [name]
        count_dataset['original'] += [len_dataset]
        count_dataset['transformed'] +=[len_transformed]

        transformed_dataset += [transformed_subset]

    transformed_dataset = ConcatDataset(transformed_dataset)
    return transformed_dataset, count_dataset