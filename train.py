import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from configs.config import gen_args
from tqdm import tqdm
from datasets.dataset import DoughDataset
from utils.robocraft_utils import prepare_input, get_scene_info, get_env_group
from metrics.metric import ChamferLoss, EarthMoverLoss, HausdorffLoss
from utils.optim import get_lr, count_parameters, my_collate, AverageMeter, Tee
from utils.utils import set_seed, matched_motion


def main():

    ### processing data ###
    phases = ['train'] if args.valid == 0 else ['valid']
    








    pass



if __name__ == '__main__':
    args = gen_args()
    set_seed(args.random_seed)
    os.system('mkdir -p ' + args.dataf)
    os.system('mkdir -p ' + args.outf)

    tee = Tee(os.path.join(args.outf, 'train.log'), 'w')


    main()


