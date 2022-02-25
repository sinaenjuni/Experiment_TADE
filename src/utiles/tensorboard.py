import os
from torch.utils.tensorboard import SummaryWriter

# TensorBoard define
def getTensorboard(log_dir):
    # log_dir = BASE_PATH + 'tb_logs/' + name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb = SummaryWriter(log_dir=log_dir)
    return tb
