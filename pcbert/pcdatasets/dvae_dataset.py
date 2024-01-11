import os
import torch
import numpy as np
import torch.utils.data as data
import glob
from pdb import set_trace
from torch.utils.data import Dataset
import h5py
import argparse


def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data

def get_scene_info(data):
    """
    A subset of prepare_input() just to get number of particles
    for initialization of grouping
    """
    positions, shape_quats, scene_params = data
    n_shapes = shape_quats.shape[0]
    count_nodes = positions.shape[0]
    n_particles = count_nodes - n_shapes

    return n_particles, n_shapes, scene_params


class ShapeNet(data.Dataset):
    def __init__(self, args, phase):
        self.args = args
        self.phase = phase
        self.data_names = ['positions', 'shape_quats', 'scene_params']
        self.dataf = os.path.join(self.args.dataf, phase)
        self.tmp_dataf_list = glob.glob(os.path.join(self.dataf, "*", "*", "*.h5"))
        self.shape_dataf_list = []
        for data in self.tmp_dataf_list:
            if "gt" not in data:
                self.shape_dataf_list.append(data)

        # data = load_data(self.data_names, data_path)
        self.n_particle, self.n_shape, self.scene_params = get_scene_info(load_data(
            self.data_names, self.shape_dataf_list[0]))
        self.permutation = np.arange(self.n_particle)
    
    def __len__(self):
        return len(self.shape_dataf_list)


    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
        
    def __getitem__(self, idx):
        # set_trace()

        data_path = self.shape_dataf_list[idx]
        data = load_data(self.data_names, data_path)
        pcs = data[0][:self.n_particle]
        pcs = self.random_sample(pcs, self.n_particle)
        pcs = self.pc_norm(pcs)

        return pcs

        # print(pcs.shape)
        # print(self.n_particle)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataf", type=str, default="/nvme/tianyang")
    args = parser.parse_args()

    toy_dataset = ShapeNet(args, phase="sample_pybert_data")
    for data in toy_dataset:
        print(data.shape)



