import torch
import numpy as np
from torch.utils.data import Sampler

class SelectSampler(Sampler):
    def __init__(self, data_source, target_label, shuffle):
        targets = torch.tensor(data_source.targets)
        self.target_idx = np.where(targets == target_label)[0]
        self.shuffle = shuffle
        self.n = len(self.target_idx)
    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            yield from self.target_idx[torch.randperm(self.n, generator=generator).tolist()]
        else:
            yield from self.target_idx

    def __len__(self):
        return self.n
