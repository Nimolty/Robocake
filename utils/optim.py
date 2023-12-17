import cv2
import glob
import h5py
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import open3d as o3d
import os
import random
import scipy
import sys
import time
import torch

from collections import defaultdict
from itertools import product
from sklearn.cluster import KMeans
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim


######################## Neural Network Regarding Trainig #######################
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def my_collate(batch):
    len_batch = len(batch[0])
    len_rel = 3

    ret = []
    for i in range(len_batch - len_rel - 1):
        d = [item[i] for item in batch]
        if isinstance(d[0], int):
            d = torch.LongTensor(d)
        else:
            d = torch.FloatTensor(torch.stack(d))
        ret.append(d)

    # processing relations
    # R: B x seq_length x n_rel x (n_p + n_s)
    for i in range(len_rel):
        R = [item[-len_rel + i - 1] for item in batch]
        max_n_rel = 0
        seq_length, _, N = R[0].size()
        for j in range(len(R)):
            max_n_rel = max(max_n_rel, R[j].size(1))
        for j in range(len(R)):
            r = R[j]
            r = torch.cat([r, torch.zeros(seq_length, max_n_rel - r.size(1), N)], 1)
            R[j] = r

        R = torch.FloatTensor(torch.stack(R))

        ret.append(R)

    # std reg
    d = [item[-1] for item in batch]
    if d[0] is not None:
        if isinstance(d[0], int):
            d = torch.LongTensor(d)
        else:
            d = torch.FloatTensor(torch.stack(d))
        ret.append(d)
    else:
        ret.append(None)
    return tuple(ret)


def get_optimizer(params, optimizer_mode, lr, beta1, beta2=0.999, momentum=0.9):
    """Returns an optimizer object."""
    if optimizer_mode == 'Adam':
        optimizer = optim.Adam(params, lr=lr, betas=(beta1, beta2))  
    elif optimizer_mode == "SGD":
        optimizer = optim.SGD(params, lr=lr, momentum=momentum)
    else:
        raise NotImplementedError(
        f'Optimizer {optimizer_mode} not supported yet!')

    return optimizer


######################## recording and loading ########################
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()

    def close(self):
        self.__del__()

