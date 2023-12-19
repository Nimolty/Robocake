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
from tqdm.notebook import tqdm
import glob
from datetime import datetime
from pdb import set_trace

from render_data_utils import update_camera, set_parameters, update_primitive, save_files, random_rotate
from render_data_utils import random_pose, get_obs, select_tool, random_materials, save_parameters


########################## render shape control example data ##########################
def render_example():
    task_name = 'ngrip'
    env_type = '_fixed'

    # gripper_fixed.yml
    cfg = load(f"../plb/envs/gripper{env_type}.yml") 
    print(cfg)
    env = TaichiEnv(cfg, nn=False, loss=False)
    env.initialize()
    state = env.get_state()

    env.set_state(**state)
    taichi_env = env
    print(env.renderer.camera_pos)
    env.renderer.camera_pos[0] = 0.5
    env.renderer.camera_pos[1] = 2.5
    env.renderer.camera_pos[2] = 0.5
    env.renderer.camera_rot = (1.57, 0.0)

    env.primitives.primitives[0].set_state(0, [0.3, 0.4, 0.5, 1, 0, 0, 0])
    env.primitives.primitives[1].set_state(0, [0.7, 0.4, 0.5, 1, 0, 0, 0])

    rgb_img, depth_img = env.render(mode="img")

    action_dim = taichi_env.primitives.action_dim
    cv2.imwrite(f"./example.png", rgb_img[..., ::-1])

    # cwd = os.getcwd()
    # root_dir = cwd + "/../.."
    # print(f'root: {root_dir}')

    # task_params = {
    #     "mid_point": np.array([0.5, 0.14, 0.5, 0, 0, 0]),
    #     "sample_radius": 0.4,
    #     "len_per_grip": 30,
    #     "len_per_grip_back": 10,
    #     "floor_pos": np.array([0.5, 0, 0.5]q),
    #     "n_shapes": 3, 
    #     "n_shapes_floor": 9,
    #     "n_shapes_per_gripper": 11,
    #     "gripper_mid_pt": int((11 - 1) / 2),
    #     "gripper_rate_limits": np.array([0.14, 0.06]), # ((0.4 * 2 - (0.23)) / (2 * 30), (0.4 * 2 - 0.15) / (2 * 30)),
    #     "p_noise_scale": 0.01,
    # }

    # if env_type == '':
    #     task_params["p_noise_scale"] = 0.03

    # print(f'p_noise_scale: {task_params["p_noise_scale"]}')


########################## render shape control single-physics data ##########################
def render_single_physics_data():
    task_name = 'ngrip'
    env_type = '_fixed'

    # gripper_fixed.yml
    cfg = load(f"../plb/envs/gripper{env_type}.yml") 
    print(cfg)
    env = TaichiEnv(cfg, nn=False, loss=False)
    env.initialize()
    state = env.get_state()

    env.set_state(**state)

    cwd = os.getcwd()
    root_dir = cwd + "/.."
    print(f'root: {root_dir}')

    task_params = {
        "mid_point": np.array([0.5, 0.14, 0.5, 0, 0, 0]),
        "sample_radius": 0.4,
        "len_per_grip": 30,
        "len_per_grip_back": 10,
        "floor_pos": np.array([0.5, 0, 0.5]),
        "n_shapes": 3, 
        "n_shapes_floor": 9,
        "n_shapes_per_gripper": 11,
        "gripper_mid_pt": int((11 - 1) / 2),
        "gripper_rate_limits": np.array([0.14, 0.06]), # ((0.4 * 2 - (0.23)) / (2 * 30), (0.4 * 2 - 0.15) / (2 * 30)),
        "p_noise_scale": 0.01,
    }

    if env_type == '':
        task_params["p_noise_scale"] = 0.03


    i = 0
    n_vid = 5
    suffix = ''
    n_grips = 3
    zero_pad = np.array([0,0,0])

    time_now = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
    rollout_dir = f"{root_dir}/dataset/{task_name}{env_type}{suffix}_{time_now}"

    while i < n_vid: 
        print(f"+++++++++++++++++++{i}+++++++++++++++++++++")
        env.set_state(**state)
        taichi_env = env    
        update_camera(env)
        set_parameters(env, yield_stress=200, E=5e3, nu=0.2) # 200， 5e3, 0.2 # 300, 800, 0.2
        update_primitive(env, [0.3, 0.4, 0.5, 1, 0, 0, 0], [0.7, 0.4, 0.5, 1, 0, 0, 0])
        save_files(env, root_dir, task_name, rollout_dir, i)
        action_dim = env.primitives.action_dim
        imgs = [] 
        true_idx = 0
        for k in range(n_grips):
            print(k)
            prim1, prim2, cur_angle = random_pose(task_name, task_params)
            update_primitive(env, prim1, prim2)
            if 'small' in suffix:
                tool_size = 0.025
            else:
                tool_size = 0.045
            select_tool(env, tool_size)
            
            gripper_rate_limit = [(task_params['sample_radius'] * 2 - (task_params['gripper_rate_limits'][0] + 2 * tool_size)) / (2 * task_params['len_per_grip']),
                                (task_params['sample_radius'] * 2 - (task_params['gripper_rate_limits'][1] + 2 * tool_size)) / (2 * task_params['len_per_grip'])]
            rate = np.random.uniform(*gripper_rate_limit)
            actions = []
            counter = 0 
            mid_point = (prim1[:3] + prim2[:3]) / 2
            prim1_direction = mid_point - prim1[:3]
            prim1_direction = prim1_direction / np.linalg.norm(prim1_direction)
            while counter < task_params["len_per_grip"]:
                prim1_action = rate * prim1_direction
                actions.append(np.concatenate([prim1_action/0.02, zero_pad, -prim1_action/0.02, zero_pad]))
                counter += 1
            counter = 0
            while counter < task_params["len_per_grip_back"]:
                prim1_action = -rate * prim1_direction
                actions.append(np.concatenate([prim1_action/0.02, zero_pad, -prim1_action/0.02, zero_pad]))
                counter += 1

            actions = np.stack(actions)
                
            for idx, act in enumerate(tqdm(actions, total=actions.shape[0])):
                env.step(act)
                obs = get_obs(env, 300)
                x = obs[0][:300]
                
                primitive_state = [env.primitives.primitives[0].get_state(0), env.primitives.primitives[1].get_state(0)]

                img = env.render_multi(mode='rgb_array', spp=3)
                rgb, depth = img[0], img[1]

                os.system('mkdir -p ' + f"{rollout_dir}/{i:03d}")
                
                for num_cam in range(4):
                    cv2.imwrite(f"{rollout_dir}/{i:03d}/{true_idx:03d}_rgb_{num_cam}.png", rgb[num_cam][..., ::-1])
                with open(f"{rollout_dir}/{i:03d}/{true_idx:03d}_depth_prim.npy", 'wb') as f:
                    np.save(f, depth + primitive_state + [tool_size])
                with open(f"{rollout_dir}/{i:03d}/{true_idx:03d}_gtp.npy", 'wb') as f:
                    np.save(f, x)
                true_idx += 1

            print(true_idx)
        
        os.system(f'ffmpeg -y -i {rollout_dir}/{i:03d}/%03d_rgb_0.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {rollout_dir}/{i:03d}/vid{i:03d}.mp4')
        i += 1


########################## render shape control multi-physics data ##########################
def render_multi_physics_data(
                            n_vid=100, E_low=1000, E_upper=8000, E_interval=50, nu_low=0.2, nu_upper=0.4, nu_interval=0.05,
                            yield_stress=200, task_name = 'ngrip', env_type = '_fixed'
                            ):
    # gripper_fixed.yml
    cfg = load(f"../plb/envs/gripper{env_type}.yml") 
    yield_stress_selected = yield_stress
    # print(cfg.SIMULATOR)
    # set_trace()


    # cfg.defrost()
    # cfg.SIMULATOR.E = 8000
    # cfg.freeze()
    # print(cfg.SIMULATOR)
    # set_trace()   

    env = TaichiEnv(cfg, nn=False, loss=False)
    env.initialize()
    state = env.get_state()
    env.set_state(**state)
    # set_trace()

    cwd = os.getcwd()
    root_dir = cwd + "/.."
    print(f'root: {root_dir}')

    task_params = {
        "mid_point": np.array([0.5, 0.14, 0.5, 0, 0, 0]),
        "sample_radius": 0.4,
        "len_per_grip": 30,
        "len_per_grip_back": 10,
        "floor_pos": np.array([0.5, 0, 0.5]),
        "n_shapes": 3, 
        "n_shapes_floor": 9,
        "n_shapes_per_gripper": 11,
        "gripper_mid_pt": int((11 - 1) / 2),
        "gripper_rate_limits": np.array([0.14, 0.06]), # ((0.4 * 2 - (0.23)) / (2 * 30), (0.4 * 2 - 0.15) / (2 * 30)),
        "p_noise_scale": 0.01,
    }

    if env_type == '':
        task_params["p_noise_scale"] = 0.03
    # set_trace()

    i = 0
    # n_vid = 100
    suffix = ''
    n_grips = 3
    zero_pad = np.array([0,0,0])

    time_now = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
    rollout_dir = f"{root_dir}/dataset/{task_name}{env_type}{suffix}_{time_now}"   
    for i in tqdm(range(n_vid)): 
        print(f"+++++++++++++++++++{i}+++++++++++++++++++++")
        env.set_state(**state)
        taichi_env = env    
        update_camera(env)
        
        E_selected, nu_selected = random_materials(E_low, E_upper, E_interval, nu_low, nu_upper, nu_interval)
        physics_params = {"yield_stress": yield_stress_selected, "E": E_selected, "nu" : nu_selected}
        print(physics_params)
        set_parameters(env, yield_stress=yield_stress_selected, 
                       E=E_selected, nu=nu_selected) # randomize
        save_parameters(physics_params, root_dir, task_name, rollout_dir, i)

        update_primitive(env, [0.3, 0.4, 0.5, 1, 0, 0, 0], [0.7, 0.4, 0.5, 1, 0, 0, 0])
        save_files(env, root_dir, task_name, rollout_dir, i)
        action_dim = env.primitives.action_dim
        imgs = [] 
        true_idx = 0
        for k in range(n_grips):
            # print(k)
            prim1, prim2, cur_angle = random_pose(task_name, task_params)
            update_primitive(env, prim1, prim2)
            if 'small' in suffix:
                tool_size = 0.025
            else:
                tool_size = 0.045
            select_tool(env, tool_size)
            
            gripper_rate_limit = [(task_params['sample_radius'] * 2 - (task_params['gripper_rate_limits'][0] + 2 * tool_size)) / (2 * task_params['len_per_grip']),
                                (task_params['sample_radius'] * 2 - (task_params['gripper_rate_limits'][1] + 2 * tool_size)) / (2 * task_params['len_per_grip'])]
            rate = np.random.uniform(*gripper_rate_limit)
            actions = []
            counter = 0 
            mid_point = (prim1[:3] + prim2[:3]) / 2
            prim1_direction = mid_point - prim1[:3]
            prim1_direction = prim1_direction / np.linalg.norm(prim1_direction)
            while counter < task_params["len_per_grip"]:
                prim1_action = rate * prim1_direction
                actions.append(np.concatenate([prim1_action/0.02, zero_pad, -prim1_action/0.02, zero_pad]))
                counter += 1
            counter = 0
            while counter < task_params["len_per_grip_back"]:
                prim1_action = -rate * prim1_direction
                actions.append(np.concatenate([prim1_action/0.02, zero_pad, -prim1_action/0.02, zero_pad]))
                counter += 1

            actions = np.stack(actions)
                
            for idx, act in enumerate(actions):
                env.step(act)
                obs = get_obs(env, 300)
                x = obs[0][:300]
                
                primitive_state = [env.primitives.primitives[0].get_state(0), env.primitives.primitives[1].get_state(0)]

                img = env.render_multi(mode='rgb_array', spp=3)
                rgb, depth = img[0], img[1]

                os.system('mkdir -p ' + f"{rollout_dir}/{i:03d}")
                
                for num_cam in range(4):
                    cv2.imwrite(f"{rollout_dir}/{i:03d}/{true_idx:03d}_rgb_{num_cam}.png", rgb[num_cam][..., ::-1])
                with open(f"{rollout_dir}/{i:03d}/{true_idx:03d}_depth_prim.npy", 'wb') as f:
                    np.save(f, depth + primitive_state + [tool_size])
                with open(f"{rollout_dir}/{i:03d}/{true_idx:03d}_gtp.npy", 'wb') as f:
                    np.save(f, x)
                true_idx += 1

            print(true_idx)
        
        os.system(f'ffmpeg -y -i {rollout_dir}/{i:03d}/%03d_rgb_0.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {rollout_dir}/{i:03d}/vid{i:03d}.mp4')
        # i += 1

















if __name__ == "__main__":
    # render_example()
    # render_single_physics_data()
    render_multi_physics_data(n_vid=500, E_low=1000, E_upper=8000, E_interval=50, nu_low=0.2, nu_upper=0.4, nu_interval=0.005)
    pass    