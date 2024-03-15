import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.optim import SGD, Adam
import time
import torch


_log_path = None

def set_log_path(save_path):
    global _log_path
    _log_path = save_path

def log(obj, filename='log.txt'):
    #prints the object passed in console
    print(obj)

    #prints into the log file
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_save_path(path, remove= True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove:
            #remove all except yaml file but the epoch models will be lost
            files  = [f for f in os.listdir(path) if not (f.endswith('.yaml') or f.endswith('epoch_last.pth'))]
            for file in files:
                os.remove(os.path.join(path, file))
    else:
        os.makedirs(path)


def set_save_path(save_path, remove = True):

    #ensure path exists by creating or removing if already existing
    ensure_save_path(save_path, remove = False)

    #setting the path to log file within save dir
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    #returning log func
    return log, writer


def make_optimizer(param, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param, **optimizer_spec['args'])

    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.2f}M'.format(tot / 1e6)
        else:
            return '{:.2f}K'.format(tot / 1e3)
    else:
        return tot


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v
    

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v
    

def calc_psnr(pred, out, rgb_range=1):
    diff = (pred - out)/ rgb_range
    mse = torch.mean(torch.pow(diff, 2))

    return -10 * torch.log10(mse)


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)