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
import shutil
import glob
import torch
from tqdm import tqdm

from collections import defaultdict
from itertools import product
from sklearn.cluster import KMeans
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch import distributed as dist

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def matched_motion(p_cur, p_prev, n_particles):
    x = p_cur[:, :n_particles, :]
    y = p_prev[:, :n_particles, :]

    x_ = x[:, :, None, :].repeat(1, 1, y.size(1), 1)
    y_ = y[:, None, :, :].repeat(1, y.size(1), 1, 1)
    dis = torch.norm(torch.add(x_, -y_), 2, dim=3)
    x_list = []
    y_list = []
    for i in range(dis.shape[0]):
        cost_matrix = dis[i].detach().cpu().numpy()
        ind1, ind2 = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)
        x_list.append(x[i, ind1])
        y_list.append(y[i, ind2])
    new_x = torch.stack(x_list)
    new_y = torch.stack(y_list)
    p_cur_new = torch.cat((new_x, p_cur[:, n_particles:, :]), dim=1)
    p_prev_new = torch.cat((new_y, p_prev[:, n_particles:, :]), dim=1)
    dist = torch.add(p_cur_new, -p_prev_new)
    return dist


##################### load and save models #####################
def load_checkpoint(save_path, device):
    checkpoint = torch.load(save_path, map_location=device)
    return checkpoint


def save_checkpoint(epoch, model, optimizer, step, save_path):
    save_dict = {
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'step': step,
                }
    torch.save(save_dict, save_path)


def load_model(model, load_model_state_dict):
    state_dict = {}
    for k in load_model_state_dict:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k] = load_model_state_dict[k]
        else:
            state_dict[f"module.{k}"] = load_model_state_dict[k]
    model_state_dict = model.state_dict()

    for k in model_state_dict:
      if not (k in state_dict):
        print('No param {}.'.format(k))
        state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=True)
    return model


def load_single_model(model, load_model_state_dict):
    state_dict = {}
    for k in load_model_state_dict:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = load_model_state_dict[k]
        else:
            state_dict[k] = load_model_state_dict[k]
    model_state_dict = model.state_dict()
    for k in model_state_dict:
      if not (k in state_dict):
        print('No param {}.'.format(k))
        state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=True)
    return model   


##################### io #####################
def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True  


#################### parallel training ####################
def reduce_mean(losses, num_gpus):
    rt = losses.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= num_gpus
    return rt


def move(source_root_dir, target_root_dir, max_num):
    exists_or_mkdir(target_root_dir)
    E_dict = {"E_00500_02000": {"interval": [500, 2000], "num":0},
              # "E_03000_06000": {"interval": [3000, 6000], "num":0},
              "E_07000_10000": {"interval": [7000, 10000], "num":0},
                }
    
    source_path_list = glob.glob(os.path.join(source_root_dir, "*"))
    for source_path in tqdm(source_path_list):
        this_path_list = glob.glob(os.path.join(source_path, "*"))
        if len(this_path_list) != 240:
            continue
        else:
            physics_params = np.load(os.path.join(source_path, "physics_params.npy"), allow_pickle=True).item()
            this_E = physics_params["E"]
            for key in E_dict:
                lower, upper = E_dict[key]["interval"]
                curr_num = E_dict[key]["num"]
                if (lower <= this_E <= upper) and curr_num < max_num:
                    E_dict[key]["num"] += 1
                    destination_folder = os.path.join(target_root_dir, key)
                    exists_or_mkdir(destination_folder)
                    shutil.move(source_path, destination_folder)



if __name__ == "__main__":
    source_root_dir = f"/nvme/tianyang/sample_Robocake_data"
    target_root_dir = f"/nvme/tianyang/split_test_Robocake_data"
    move(source_root_dir, target_root_dir, max_num=8)