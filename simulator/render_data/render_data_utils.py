# shared across tasks
from plb.optimizer.optim import Adam
from plb.engine.taichi_env import TaichiEnv
from plb.config.default_config import get_cfg_defaults, CN

import os
import cv2
import numpy as np
import taichi as ti
ti.init(arch=ti.gpu)
import matplotlib.pyplot as plt
from plb.config import load
from tqdm import tqdm
import glob
from datetime import datetime
from transforms3d.quaternions import mat2quat
from transforms3d.axangles import axangle2mat

def main():
    pass

def set_parameters(env: TaichiEnv, yield_stress, E, nu):
    env.simulator.yield_stress.fill(yield_stress)
    _mu, _lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
    env.simulator.mu.fill(_mu)
    env.simulator.lam.fill(_lam)


def update_camera(env):
    env.renderer.camera_pos[0] = 0.5
    env.renderer.camera_pos[1] = 2.5
    env.renderer.camera_pos[2] = 0.5
    env.renderer.camera_rot = (1.57, 0.0)
    env.render_cfg.defrost()
    env.render_cfg.camera_pos_1 = (0.5, 2.5, 2.2)
    env.render_cfg.camera_rot_1 = (0.8, 0.)
    env.render_cfg.camera_pos_2 = (2.4, 2.5, 0.2)
    env.render_cfg.camera_rot_2 = (0.8, 1.8)
    env.render_cfg.camera_pos_3 = (-1.9, 2.5, 0.2)
    env.render_cfg.camera_rot_3 = (0.8, -1.8)
    env.render_cfg.camera_pos_4 = (0.5, 2.5, -1.8)
    env.render_cfg.camera_rot_4 = (0.8, 3.14)


def update_primitive(env, prim1_list, prim2_list):
    env.primitives.primitives[0].set_state(0, prim1_list)
    env.primitives.primitives[1].set_state(0, prim2_list)

def save_parameters(physics_params, root_dir, task_name, rollout_dir, i):
    os.makedirs(f"{rollout_dir}/{i:03d}", exist_ok=True)
    with open(f"{rollout_dir}/{i:03d}"+"/physics_params.npy", 'wb') as f:
        np.save(f, physics_params)


def save_files(env, root_dir, task_name, rollout_dir, i):
    files = glob.glob(f"{root_dir}/dataset/{task_name}/{i:03d}/*")
    for f in files:
        os.remove(f)
        
    os.makedirs(f"{rollout_dir}/{i:03d}", exist_ok=True)
    with open(f"{rollout_dir}/{i:03d}"+"/cam_params.npy", 'wb') as f:
        ext1=env.renderer.get_ext(env.render_cfg.camera_rot_1, np.array(env.render_cfg.camera_pos_1))
        ext2=env.renderer.get_ext(env.render_cfg.camera_rot_2, np.array(env.render_cfg.camera_pos_2))
        ext3=env.renderer.get_ext(env.render_cfg.camera_rot_3, np.array(env.render_cfg.camera_pos_3))
        ext4=env.renderer.get_ext(env.render_cfg.camera_rot_4, np.array(env.render_cfg.camera_pos_4))
        intrinsic = env.renderer.get_int()
        cam_params = {'cam1_ext': ext1, 'cam2_ext': ext2, 'cam3_ext': ext3, 'cam4_ext': ext4, 'intrinsic': intrinsic}
        np.save(f, cam_params)

def random_rotate(mid_point, gripper1_pos, gripper2_pos, z_vec):
    mid_point = mid_point[:3]
    z_angle = np.random.uniform(0, np.pi)
    z_mat = axangle2mat(z_vec, z_angle, is_normalized=True)
    all_mat = z_mat
    quat = mat2quat(all_mat)
    return gripper1_pos, gripper2_pos, quat


def random_pose(task_name, task_params):
    p_noise_x = task_params["p_noise_scale"] * (np.random.randn() * 2 - 1)
    p_noise_z = task_params["p_noise_scale"] * (np.random.randn() * 2 - 1)
    if task_name == 'ngrip' or task_name == 'ngrip_3d':
        p_noise = np.clip(np.array([p_noise_x, 0, p_noise_z]), a_min=-0.1, a_max=0.1)
    else:
        raise NotImplementedError
    
    new_mid_point = task_params["mid_point"][:3] + p_noise

    rot_noise = np.random.uniform(0, np.pi)

    x1 = new_mid_point[0] - task_params["sample_radius"] * np.cos(rot_noise)
    z1 = new_mid_point[2] + task_params["sample_radius"] * np.sin(rot_noise)
    x2 = new_mid_point[0] + task_params["sample_radius"] * np.cos(rot_noise)
    z2 = new_mid_point[2] - task_params["sample_radius"] * np.sin(rot_noise)
    y = new_mid_point[1]
    z_vec = np.array([np.cos(rot_noise), 0, np.sin(rot_noise)])
    if task_name == 'ngrip':
        gripper1_pos = np.array([x1, y, z1])
        gripper2_pos = np.array([x2, y, z2])
        quat = np.array([1, 0, 0, 0])
    elif task_name == 'ngrip_3d':
        gripper1_pos, gripper2_pos, quat = random_rotate(new_mid_point, np.array([x1, y, z1]), np.array([x2, y, z2]), z_vec)
    else:
        raise NotImplementedError
    return np.concatenate([gripper1_pos, quat]), np.concatenate([gripper2_pos, quat]), rot_noise

def random_materials(E_low, E_upper, E_interval, nu_low, nu_upper, nu_interval):
    # E = np.random.uniform(E_low, E_upper)
    # nu = np.random.uniform(nu_low, nu_upper)
    E_range = np.arange(E_low, E_upper, E_interval)
    nu_range = np.arange(nu_low, nu_upper, nu_interval)
    E_selected = np.random.choice(E_range)
    nu_selected = np.random.choice(nu_range)
    return E_selected, nu_selected 


def get_obs(env, n_particles, t=0):
    x = env.simulator.get_x(t)
    v = env.simulator.get_v(t)
    step_size = len(x) // n_particles
    return x[::step_size], v[::step_size]


def select_tool(env, width):
    env.primitives.primitives[0].r[None] = width
    env.primitives.primitives[1].r[None] = width


if __name__ == "__main__":
    main()
    pass