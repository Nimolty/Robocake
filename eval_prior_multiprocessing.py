import glob
import numpy as np
import os
import torch

from configs.config import gen_args
from metrics.metric import ChamferLoss, EarthMoverLoss, HausdorffLoss
from utils.robocraft_utils import prepare_input, get_scene_info, get_env_group, load_data
from utils.utils import set_seed, exists_or_mkdir, load_checkpoint, load_single_model
from utils.optim import count_parameters, Tee
from visualize.visualize import plt_render, train_plot_curves, eval_plot_curves
from models.prior_model_distributed import Prior_Model

# parallel
import torch.multiprocessing as mp

# tqdm
from tqdm import tqdm
from pdb import set_trace

def prepare_model_and_data(args, device, use_gpu, prior_model=None):
    set_seed(args.random_seed)


    ########################## set path ##########################
    prior_epoch_name = args.resume_prior_path.split('/')[-2]
    prior_output_dir = os.path.dirname(args.outf)
    prior_eval_out_path = os.path.join(prior_output_dir, f"eval_{str(args.exp_id)}", prior_epoch_name, args.eval_data_class)
    exists_or_mkdir(os.path.join(prior_eval_out_path, "plot"))
    exists_or_mkdir(os.path.join(prior_eval_out_path, "render"))
    tee = Tee(os.path.join(prior_eval_out_path , 'eval.log'), 'w')
    data_names = args.data_names
    eval_data_class = args.eval_data_class
    
    
    ########################## create model ##########################
    if prior_model is None:
        prior_model = Prior_Model(args, device).to(device)
        if args.stage == 'dy':
            prior_checkpoint = load_checkpoint(args.resume_prior_path, device)
            prior_model = load_single_model(prior_model, prior_checkpoint['model_state_dict'])

    print("model #params: %d" % count_parameters(prior_model))

    
    ########################## load eval model and loss functions ##########################
    prior_model.eval()
    eval_data_list = glob.glob(os.path.join(args.dataf, eval_data_class, "*"))
    
    return prior_model, eval_data_list, prior_eval_out_path


def inference_after_training(prior_model, args, 
                             data_dict, prior_eval_out_path, use_gpu, device):
    emd_loss = EarthMoverLoss()
    chamfer_loss = ChamferLoss()
    h_loss = HausdorffLoss()

    p_gt = torch.cat(data_dict["p_gt"]).float()
    p_sample = torch.cat(data_dict["p_sample"]).float()
    scene_params = data_dict["scene_params"].detach().cpu().numpy()[0]
    n_particle = data_dict["n_particle"].item()
    n_shape = data_dict["n_shape"].item()
    physics_params = {k: v.item() for k, v in data_dict["physics_params"].items()}
    idx_episode = data_dict["idx_episode"].item()

    p_pred = torch.zeros(args.time_step, n_particle + n_shape, args.state_dim)
    # initialize particle grouping
    group_info = get_env_group(args, n_particle, scene_params, use_gpu=use_gpu)

    # memory: B x mem_nlayer x (n_particle + n_shape) x nf_memory
    # for now, only used as a placeholder
    try:
        memory_init = prior_model.module.init_memory(1, n_particle + n_shape)
    except:
        memory_init = prior_model.init_memory(1, n_particle + n_shape)
    loss_list = []
    # model rollout
    # loss = 0.
    # loss_raw = 0.
    # loss_counter = 0
    st_idx = args.n_his
    ed_idx = args.time_step
    with torch.no_grad():
        for step_id in tqdm(range(st_idx, ed_idx)):
            # print(step_id)
            if step_id == st_idx:
                if args.gt_particles:
                    # state_cur (unnormalized): n_his x (n_p + n_s) x state_dim
                    state_cur = p_gt[step_id - args.n_his:step_id]
                else:
                    state_cur = p_sample[step_id - args.n_his:step_id]
                state_cur = state_cur.to(device)

            # unsqueeze the batch dimension
            # attr: B x (n_p + n_s) x attr_dim
            # Rr_cur, Rs_cur: B x n_rel x (n_p + n_s)
            # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
            attr, _, Rr_cur, Rs_cur, Rn_cur, cluster_onehot = prepare_input(state_cur[-1].cpu().numpy(), n_particle,
                                                                    n_shape, args, stdreg=args.stdreg)
            attr = attr.to(device).unsqueeze(0)
            Rr_cur = Rr_cur.to(device).unsqueeze(0)
            Rs_cur = Rs_cur.to(device).unsqueeze(0)
            Rn_cur = Rn_cur.to(device).unsqueeze(0)
            state_cur = state_cur.unsqueeze(0)
            if cluster_onehot:
                cluster_onehot = cluster_onehot.unsqueeze(0)

            if args.stage in ['dy']:
                inputs = [attr, state_cur, Rr_cur, Rs_cur, Rn_cur, memory_init, group_info, cluster_onehot]

            # pred_pos (unnormalized): B x n_p x state_dim
            # pred_motion_norm (normalized): B x n_p x state_dim
            if args.sequence_length > args.n_his + 1:
                pred_pos_p, pred_motion_norm, std_cluster = prior_model(inputs, (step_id - args.n_his), prior_remove_his_particles=args.prior_remove_his_particles)
            else:
                pred_pos_p, pred_motion_norm, std_cluster = prior_model(inputs,prior_remove_his_particles=args.prior_remove_his_particles)

            # concatenate the state of the shapes
            # pred_pos (unnormalized): B x (n_p + n_s) x state_dim
            sample_pos = p_sample[step_id].to(device).unsqueeze(0)
            sample_pos_p = sample_pos[:, :n_particle]
            pred_pos = torch.cat([pred_pos_p, sample_pos[:, n_particle:]], 1)

            # sample_motion_norm (normalized): B x (n_p + n_s) x state_dim
            # pred_motion_norm (normalized): B x (n_p + n_s) x state_dim
            sample_motion = (p_sample[step_id] - p_sample[step_id - 1]).unsqueeze(0)
            sample_motion = sample_motion.to(device)

            try:
                mean_d, std_d = prior_model.stat[2:]
            except:
                mean_d, std_d = prior_model.module.stat[2:]
            sample_motion_norm = (sample_motion - mean_d) / std_d
            pred_motion_norm = torch.cat([pred_motion_norm, sample_motion_norm[:, n_particle:]], 1)

            loss_emd = emd_loss(pred_pos_p, sample_pos_p)
            loss_chamfer = chamfer_loss(pred_pos_p, sample_pos_p)
            loss_h = h_loss(pred_pos_p, sample_pos_p)

            loss_list.append([step_id, loss_emd.item(), loss_chamfer.item(), loss_h.item()])
            # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
            state_cur = torch.cat([state_cur[:, 1:], pred_pos.unsqueeze(1)], 1)
            state_cur = state_cur.detach()[0]

            # record the prediction
            p_pred[step_id] = state_cur[-1].detach().cpu()

    # loss_list_over_episodes.append(loss_list)

    # visualization
    group_info = [d.data.cpu().numpy()[0, ...] for d in group_info]

    if args.gt_particles:
        p_pred = np.concatenate((p_gt.numpy()[:st_idx], p_pred.numpy()[st_idx:ed_idx]))
    else:
        p_pred = np.concatenate((p_sample.numpy()[:st_idx], p_pred.numpy()[st_idx:ed_idx]))

    p_sample = p_sample.numpy()[:ed_idx]
    p_gt = p_gt.numpy()[:ed_idx]

    # vid_path = os.path.join(args.dataf, 'vid', str(idx_episode).zfill(3))
    render_path = os.path.join(prior_eval_out_path, 'render', f'vid_{idx_episode}_plt.gif')

    if args.vis == 'plt':
        plt_render([p_gt, p_sample, p_pred], n_particle, render_path, physics_params)
    else:
        pass
    
    return loss_list 



def inference(prior_model, this_eval_data, eval_data_class, 
              data_names, prior_eval_out_path, args, use_gpu, device):
    emd_loss = EarthMoverLoss()
    chamfer_loss = ChamferLoss()
    h_loss = HausdorffLoss()
    
    idx_episode = int(this_eval_data.split('/')[-1])
    loss_list = []

    print("Prior Rollout %d / %d" % (idx_episode, args.n_rollout))

    n_particle, n_shape = 0, 0

    # load data
    gt_data_list = []
    data_list = []
    p_gt = []
    p_sample = []
    frame_list = sorted(glob.glob(os.path.join(args.dataf, eval_data_class, str(idx_episode).zfill(3), 'shape_*.h5')))
    gt_frame_list = sorted(glob.glob(os.path.join(args.dataf, eval_data_class, str(idx_episode).zfill(3), 'shape_gt_*.h5')))
    physics_params_path = os.path.join(args.dataf, eval_data_class, str(idx_episode).zfill(3), "physics_params.npy")
    physics_params = np.load(physics_params_path, allow_pickle=True).item()
    # print(type(physics_params))
    # set_trace()

    args.time_step = (len(frame_list) - len(gt_frame_list))
    for step in range(args.time_step):
        gt_frame_name = 'gt_' + str(step) + '.h5'
        frame_name = str(step) + '.h5'
        if args.shape_aug:
            gt_frame_name = 'shape_' + gt_frame_name
            frame_name = 'shape_' + frame_name

        gt_data_path = os.path.join(args.dataf, eval_data_class, str(idx_episode).zfill(3), gt_frame_name)
        data_path = os.path.join(args.dataf, eval_data_class, str(idx_episode).zfill(3), frame_name)

        try:
            gt_data = load_data(data_names, gt_data_path)
            load_gt = True
        except FileNotFoundError:
            load_gt = False
        
        data = load_data(data_names, data_path)

        if n_particle == 0 and n_shape == 0:
            n_particle, n_shape, scene_params = get_scene_info(data)
            scene_params = torch.FloatTensor(scene_params).unsqueeze(0)

        if args.verbose_data:
            print("n_particle", n_particle)
            print("n_shape", n_shape)

        if load_gt: 
            gt_data_list.append(gt_data)
        data_list.append(data)

        if load_gt: 
            p_gt.append(gt_data[0])

        new_state = data[0]

        p_sample.append(new_state)

    # p_sample: time_step x N x state_dim
    if load_gt: 
        p_gt = torch.FloatTensor(np.stack(p_gt))
    p_sample = torch.FloatTensor(np.stack(p_sample))
    p_pred = torch.zeros(args.time_step, n_particle + n_shape, args.state_dim)
    # initialize particle grouping
    group_info = get_env_group(args, n_particle, scene_params, use_gpu=use_gpu)

    # memory: B x mem_nlayer x (n_particle + n_shape) x nf_memory
    # for now, only used as a placeholder
    memory_init = prior_model.init_memory(1, n_particle + n_shape)

    # model rollout
    # loss = 0.
    # loss_raw = 0.
    # loss_counter = 0
    st_idx = args.n_his
    ed_idx = args.time_step

    with torch.no_grad():
        for step_id in tqdm(range(st_idx, ed_idx)):
            # print(step_id)
            if step_id == st_idx:
                if args.gt_particles:
                    # state_cur (unnormalized): n_his x (n_p + n_s) x state_dim
                    state_cur = p_gt[step_id - args.n_his:step_id]
                else:
                    state_cur = p_sample[step_id - args.n_his:step_id]
                state_cur = state_cur.to(device)

            # unsqueeze the batch dimension
            # attr: B x (n_p + n_s) x attr_dim
            # Rr_cur, Rs_cur: B x n_rel x (n_p + n_s)
            # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
            attr, _, Rr_cur, Rs_cur, Rn_cur, cluster_onehot = prepare_input(state_cur[-1].cpu().numpy(), n_particle,
                                                                    n_shape, args, stdreg=args.stdreg)
            attr = attr.to(device).unsqueeze(0)
            Rr_cur = Rr_cur.to(device).unsqueeze(0)
            Rs_cur = Rs_cur.to(device).unsqueeze(0)
            Rn_cur = Rn_cur.to(device).unsqueeze(0)
            state_cur = state_cur.unsqueeze(0)
            if cluster_onehot:
                cluster_onehot = cluster_onehot.unsqueeze(0)

            if args.stage in ['dy']:
                inputs = [attr, state_cur, Rr_cur, Rs_cur, Rn_cur, memory_init, group_info, cluster_onehot]

            # pred_pos (unnormalized): B x n_p x state_dim
            # pred_motion_norm (normalized): B x n_p x state_dim
            if args.sequence_length > args.n_his + 1:
                pred_pos_p, pred_motion_norm, std_cluster = prior_model(inputs, (step_id - args.n_his), prior_remove_his_particles=args.prior_remove_his_particles)
            else:
                pred_pos_p, pred_motion_norm, std_cluster = prior_model(inputs,prior_remove_his_particles=args.prior_remove_his_particles)

            # concatenate the state of the shapes
            # pred_pos (unnormalized): B x (n_p + n_s) x state_dim
            sample_pos = p_sample[step_id].to(device).unsqueeze(0)
            sample_pos_p = sample_pos[:, :n_particle]
            pred_pos = torch.cat([pred_pos_p, sample_pos[:, n_particle:]], 1)

            # sample_motion_norm (normalized): B x (n_p + n_s) x state_dim
            # pred_motion_norm (normalized): B x (n_p + n_s) x state_dim
            sample_motion = (p_sample[step_id] - p_sample[step_id - 1]).unsqueeze(0)
            sample_motion = sample_motion.to(device)

            mean_d, std_d = prior_model.stat[2:]
            sample_motion_norm = (sample_motion - mean_d) / std_d
            pred_motion_norm = torch.cat([pred_motion_norm, sample_motion_norm[:, n_particle:]], 1)

            loss_emd = emd_loss(pred_pos_p, sample_pos_p)
            loss_chamfer = chamfer_loss(pred_pos_p, sample_pos_p)
            loss_h = h_loss(pred_pos_p, sample_pos_p)

            loss_list.append([step_id, loss_emd.item(), loss_chamfer.item(), loss_h.item()])
            # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
            state_cur = torch.cat([state_cur[:, 1:], pred_pos.unsqueeze(1)], 1)
            state_cur = state_cur.detach()[0]

            # record the prediction
            p_pred[step_id] = state_cur[-1].detach().cpu()

    # loss_list_over_episodes.append(loss_list)

    # visualization
    group_info = [d.data.cpu().numpy()[0, ...] for d in group_info]

    if args.gt_particles:
        p_pred = np.concatenate((p_gt.numpy()[:st_idx], p_pred.numpy()[st_idx:ed_idx]))
    else:
        p_pred = np.concatenate((p_sample.numpy()[:st_idx], p_pred.numpy()[st_idx:ed_idx]))

    p_sample = p_sample.numpy()[:ed_idx]
    if load_gt: 
        p_gt = p_gt.numpy()[:ed_idx]

    # vid_path = os.path.join(args.dataf, 'vid', str(idx_episode).zfill(3))
    render_path = os.path.join(prior_eval_out_path, 'render', f'vid_{idx_episode}_plt.gif')

    if args.vis == 'plt':
        plt_render([p_gt, p_sample, p_pred], n_particle, render_path, physics_params)
    else:
        pass
    
    return loss_list 

def print_eval(args, prior_eval_out_path, loss_list_over_episodes):    
    try:
        with open(os.path.join(args.outf, 'prior_train.npy'), 'rb') as f:
            train_log = np.load(f, allow_pickle=True)
            train_log = train_log[None][0]
            train_plot_curves(train_log['iters'], train_log['loss'], path=os.path.join(prior_eval_out_path, 'plot', 'train_loss_curves.png'))
    except:
        pass
    
    loss_list_over_episodes = np.array(loss_list_over_episodes)
    loss_mean = np.mean(loss_list_over_episodes, axis=0)
    loss_std = np.std(loss_list_over_episodes, axis=0)
    eval_plot_curves(loss_mean[:, :-1], loss_std[:, :-1], path=os.path.join(prior_eval_out_path, 'plot', 'eval_loss_curves.png'))

    result_dict = {}
    result_dict["Last Frame EMD"] = np.mean(loss_list_over_episodes[:, -1, 1])
    result_dict["Last Frame CD"] = np.mean(loss_list_over_episodes[:, -1, 2])
    result_dict["Last Frame HD"] = np.mean(loss_list_over_episodes[:, -1, 3])
    result_dict["Over Episodes EMD"] = np.mean(loss_list_over_episodes[:, :, 1])
    result_dict["Over Episodes CD"] = np.mean(loss_list_over_episodes[:, :, 2])
    result_dict["Over Episodes HD"] = np.mean(loss_list_over_episodes[:, :, 3])

    print(f"\nAverage emd loss at last frame: {result_dict['Last Frame EMD']} (+- {np.std(loss_list_over_episodes[:, -1, 1])})")
    print(f"Average chamfer loss at last frame: {result_dict['Last Frame CD']} (+- {np.std(loss_list_over_episodes[:, -1, 2])})")
    print(f"Average hausdorff loss at last frame: {result_dict['Last Frame HD']} (+- {np.std(loss_list_over_episodes[:, -1, 3])})")

    print(f"\nAverage emd loss over episodes: {result_dict['Over Episodes EMD']} (+- {np.std(loss_list_over_episodes[:, :, 1])})")
    print(f"Average chamfer loss over episodes: {result_dict['Over Episodes CD']} (+- {np.std(loss_list_over_episodes[:, :, 2])})")
    print(f"Average hausdorff loss over episodes: {result_dict['Over Episodes HD']} (+- {np.std(loss_list_over_episodes[:, :, 3])})")

    
    return result_dict

    
if __name__ == "__main__":
    args = gen_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = (device == torch.device("cuda"))

    args.outf = os.path.join(args.outf, str(args.exp_id))
    exists_or_mkdir(args.dataf)
    args.eval_prior_path  = args.outf 
   
    prior_model, eval_data_list, prior_eval_out_path = prepare_model_and_data(args, device, use_gpu)
    num_processes = 8
    
    # set before any multi process pool
    mp.set_start_method('spawn', force=True)
    # eval_data_

    # create pool
    pool = mp.Pool(processes=num_processes)  
    # inference(residual_model, prior_model, this_eval_data, eval_data_class, data_names, residual_eval_out_path)
    results = pool.starmap(inference, [(prior_model, eval_data, args.eval_data_class, args.data_names, 
                                        prior_eval_out_path, args, use_gpu, device) 
                                       for eval_data in eval_data_list])  
    pool.close()
    pool.join()
    loss_list_over_episodes = []

    for result in results:
        loss_list_over_episodes.append(result)
    
    print_eval(args, prior_eval_out_path, loss_list_over_episodes)   
    
    
    
    
    
    