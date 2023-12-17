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



