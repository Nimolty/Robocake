import sys 
sys.path.append("..") 
# print("this sys path: ", sys.path)

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

from utils.robocraft_utils import load_data, get_scene_info, real_sim_remap, prepare_input, calc_rigid_transform


class DoughDataset(Dataset):

    def __init__(self, args, phase):
        self.args = args
        self.phase = phase
        self.data_dir = os.path.join(self.args.dataf, phase)
        self.vision_dir = self.data_dir + '_vision'
        # self.stat_path = os.path.join(self.args.dataf, '..', 'stats.h5')

        self.n_frame_list = []
        self.vid_list = sorted(glob.glob(os.path.join(self.data_dir, '*')))
        self.vid_list.sort()
        # print(vid_list)
        for vid_idx in range(len(self.vid_list)):
            frame_list = sorted(glob.glob(os.path.join(self.vid_list[vid_idx], 'shape_*.h5')))
            gt_frame_list = sorted(glob.glob(os.path.join(self.vid_list[vid_idx], 'shape_gt_*.h5')))
            self.n_frame_list.append(len(frame_list) - len(gt_frame_list))
        print(f"#frames list: {self.n_frame_list}")

        # self.n_rollout = len

        if args.gen_data:
            os.system('mkdir -p ' + self.data_dir)
        if args.gen_vision:
            os.system('mkdir -p ' + self.vision_dir)

        if args.env in ['Gripper']:
            self.data_names = ['positions', 'shape_quats', 'scene_params']
        else:
            raise AssertionError("Unsupported env")

        ratio = self.args.train_valid_ratio
        if phase == 'train':
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase == 'valid':
            self.n_rollout = self.args.n_rollout - int(self.args.n_rollout * ratio)
        else:
            raise AssertionError("Unknown phase")

    def __len__(self):
        """
        Each data point is consisted of a whole trajectory
        """
        args = self.args
        return len(self.vid_list) * (args.time_step - args.sequence_length + 1)

    # def load_data(self, name):
    #     print("Loading stat from %s ..." % self.stat_path)
    #     self.stat = load_data(self.data_names[:1], self.stat_path)

    def __getitem__(self, idx):
        """
        Load a trajectory of length sequence_length
        """
        args = self.args

        idx_curr = idx
        idx_rollout = 0
        offset = self.n_frame_list[idx_rollout] - args.sequence_length + 1
        while idx_curr >= offset:
            idx_curr -= offset
            idx_rollout = (idx_rollout + 1) % len(self.n_frame_list)
            offset = self.n_frame_list[idx_rollout] - args.sequence_length + 1
        
        # offset = args.time_step - args.sequence_length + 1
        # idx_rollout = idx // offset
        # st_idx = idx % offset
        st_idx = idx_curr
        ed_idx = st_idx + args.sequence_length

        if args.stage in ['dy']:
            # load ground truth data
            attrs, particles, Rrs, Rss, Rns, cluster_onehots= [], [], [], [], [], []
            # sdf_list = []
            max_n_rel = 0
            for t in range(st_idx, ed_idx):
                # load data
                if self.args.env == 'Pinch' or self.args.env == 'Gripper':
                    frame_name = str(t) + '.h5'

                    if self.args.gt_particles:
                        frame_name = 'gt_' + frame_name
                    if self.args.shape_aug:
                        frame_name = 'shape_' + frame_name
                        # data_path = os.path.join(self.data_dir, str(idx_rollout).zfill(3), 'gt_' + frame_name)
                    # else:
                    #     pass
                        # data_path = os.path.join(self.data_dir, str(idx_rollout).zfill(3), str(t) + '.h5')
                    # data_path = os.path.join(self.data_dir, str(idx_rollout).zfill(3), frame_name)
                    data_path = os.path.join(self.vid_list[idx_rollout], frame_name)
                else:
                    data_path = os.path.join(self.data_dir, str(idx_rollout), str(t) + '.h5')
                data = load_data(self.data_names, data_path)
                # sdf_data = load_data(['sdf'], os.path.join(self.data_dir, str(idx_rollout).zfill(3), 'sdf_' + str(t) + '.h5')), 

                # load scene param
                if t == st_idx:
                    n_particle, n_shape, scene_params = get_scene_info(data)

                # attr: (n_p + n_s) x attr_dim
                # particle (unnormalized): (n_p + n_s) x state_dim
                # Rr, Rs: n_rel x (n_p + n_s)               
                assert "grip" in args.data_type
                new_state = data[0]

                attr, particle, Rr, Rs, Rn, cluster_onehot = prepare_input(new_state, n_particle, n_shape, self.args, stdreg=self.args.stdreg)
                max_n_rel = max(max_n_rel, Rr.size(0))

                attrs.append(attr)
                particles.append(particle.numpy())
                Rrs.append(Rr)
                Rss.append(Rs)
                Rns.append(Rn)
                # sdf_data = np.array(sdf_data).squeeze()
                # print(np.array(sdf_data.shape)
                # sdf_list.append(sdf_data)

                if cluster_onehot is not None:
                    cluster_onehots.append(cluster_onehot)


        '''
        add augmentation
        '''
        if args.stage in ['dy']:
            for t in range(args.sequence_length):
                if t == args.n_his - 1:
                    # set anchor for transforming rigid objects
                    particle_anchor = particles[t].copy()

                if t < args.n_his:
                    # add noise to observation frames - idx smaller than n_his
                    noise = np.random.randn(n_particle, 3) * args.std_d * args.augment_ratio
                    particles[t][:n_particle] += noise

                else:
                    pass

        else:
            AssertionError("Unknown stage %s" % args.stage)


        # attr: (n_p + n_s) x attr_dim
        # particles (unnormalized): seq_length x (n_p + n_s) x state_dim
        # scene_params: param_dim
        # sdf_list: seq_length x 64 x 64 x 64
        attr = torch.FloatTensor(attrs[0])
        particles = torch.FloatTensor(np.stack(particles))
        scene_params = torch.FloatTensor(scene_params)
        # sdf_list = torch.FloatTensor(np.stack(sdf_list))

        # pad the relation set
        # Rr, Rs: seq_length x n_rel x (n_p + n_s)
        if args.stage in ['dy']:
            for i in range(len(Rrs)):
                Rr, Rs, Rn = Rrs[i], Rss[i], Rns[i]
                Rr = torch.cat([Rr, torch.zeros(max_n_rel - Rr.size(0), n_particle + n_shape)], 0)
                Rs = torch.cat([Rs, torch.zeros(max_n_rel - Rs.size(0), n_particle + n_shape)], 0)
                Rn = torch.cat([Rn, torch.zeros(max_n_rel - Rn.size(0), n_particle + n_shape)], 0)
                Rrs[i], Rss[i], Rns[i] = Rr, Rs, Rn
            Rr = torch.FloatTensor(np.stack(Rrs))
            Rs = torch.FloatTensor(np.stack(Rss))
            Rn = torch.FloatTensor(np.stack(Rns))
            if cluster_onehots:
                cluster_onehot = torch.FloatTensor(np.stack(cluster_onehots))
            else:
                cluster_onehot = None
        if args.stage in ['dy']:
            return attr, particles, n_particle, n_shape, scene_params, Rr, Rs, Rn, cluster_onehot