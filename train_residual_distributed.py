### load preliminary ###
import os
import numpy as np

### load torch ###
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

### load utils ###
from configs.config import gen_args
from tqdm import tqdm
from datasets.dataset import DoughDataset
from utils.robocraft_utils import prepare_input, get_scene_info, get_env_group
from metrics.metric import ChamferLoss, EarthMoverLoss, HausdorffLoss
from utils.optim import get_lr, count_parameters, my_collate, AverageMeter, Tee, get_optimizer, distributed_concat
from utils.utils import set_seed, matched_motion, load_checkpoint, save_checkpoint, exists_or_mkdir, reduce_mean, load_model
from visualize.visualize import plt_render, plt_render_image_split
from pdb import set_trace

### load model ###
from models.prior_model_distributed import Prior_Model
from models.residual_model_distributed import Residual_Model

### parallel training ###
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
import wandb
import socket


def main(args):
    ########################## set local rank ##########################
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device=torch.device("cuda",args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        use_gpu = True
    else:
        raise NotImplementedError
    ##########################      wandb      ##########################
    if dist.get_rank() == 0:
        run_dir = os.path.join(args.run_dir, args.experiment_name, str(args.exp_id))
        exists_or_mkdir(run_dir)
        wandb.init(config=args,
                   project=args.project_name,
                   entity=args.team_name,
                   notes=socket.gethostname(),
                   name=args.experiment_name+"_"+str(args.exp_id),
                   group=args.experiment_name+"_"+str(args.exp_id),
                   dir=run_dir,
                   job_type="training",
                   reinit=True)

    ########################## processing data ##########################
    phases = ['train'] if args.valid == 0 else ['train', 'valid']
    datasets = {phase: DoughDataset(args, phase) for phase in phases}
    samplers = {phase: DistributedSampler(datasets[phase]) for phase in phases}

    # for phase in phases:
    #     datasets[phase].load_data(args.env)

    print(f"Train dataset size: {len(datasets['train'])}")
    dataloaders = {phase: DataLoader(
        datasets[phase],
        sampler=samplers[phase],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=my_collate) for phase in phases} # TODO: understand the logics of my_collate

    ########################## create model ##########################
    prior_model = Prior_Model(args, device).to(device)
    print("prior model #params: %d" % count_parameters(prior_model))
    residual_model = Residual_Model(args, device).to(device)
    print("residual model #params: %d" % count_parameters(residual_model))
    prior_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(prior_model).to(device)
    prior_model = torch.nn.parallel.DistributedDataParallel(prior_model, device_ids=[args.local_rank],
                                                output_device=args.local_rank,find_unused_parameters=True)
    residual_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(residual_model).to(device)
    residual_model = torch.nn.parallel.DistributedDataParallel(residual_model, device_ids=[args.local_rank],
                                                output_device=args.local_rank,find_unused_parameters=True)

    ########################## load pretrained model ##########################
    if args.resume_prior_path:
        print("Loading saved prior ckpt from %s" % args.resume_prior_path)
        if args.stage == 'dy':
            prior_checkpoint = load_checkpoint(args.resume_prior_path, device)
            prior_model = load_model(prior_model, prior_checkpoint['model_state_dict'])
    
    if args.resume_residual_path:
        print("Loading saved residual ckpt from %s" % args.resume_residual_path)  
        if args.stage == 'dy':
            residual_checkpoint = load_checkpoint(args.resume_residual_path, device)
            residual_model = load_model(residual_model, residual_checkpoint['model_state_dict'])
    

    num_gpus = torch.cuda.device_count()

    ########################## create optimizer ##########################
    if args.stage == 'dy':
        residual_params = residual_model.parameters()
    else:
        raise AssertionError("unknown stage: %s" % args.stage)
    
    residual_optimizer = get_optimizer(params=residual_params, optimizer_mode=args.optimizer, lr=args.lr, beta1=args.beta1)
    # reduce learning rate when a metric has stopped improving
    scheduler = ReduceLROnPlateau(residual_optimizer, 'min', factor=0.8, patience=3, verbose=True)

    if args.resume_residual_path:
        if args.stage == 'dy':
            residual_optimizer.load_state_dict(residual_checkpoint['optimizer_state_dict'])
    
    ########################## define loss ##########################
    chamfer_loss = ChamferLoss()
    emd_loss = EarthMoverLoss()
    h_loss = HausdorffLoss()


    ########################## start training ##########################
    residual_start_epoch = 0
    if args.resume_residual_path:
        if args.stage == 'dy':
            residual_start_epoch = residual_checkpoint['epoch']

    residual_best_valid_loss = np.inf
    residual_training_stats = {'args':vars(args), 'loss':[], 'loss_raw':[], 'iters': [], 'loss_emd': [], 'loss_motion': []}
    residual_rollout_epoch = -1
    residual_rollout_iter = -1
    residual_total_step = 0
    if args.resume_residual_path:
        if args.stage == 'dy':
            # residual_total_step = residual_checkpoint['step']  
            residual_total_step = residual_checkpoint["epoch"] * (int(datasets["train"].__len__()) / args.batch_size / num_gpus)   

    for residual_epoch in range(residual_start_epoch, args.residual_n_epoch):
        for phase in phases:
            samplers[phase].set_epoch(residual_epoch)
            print("phase", phase)
            print("epoch", residual_epoch)
            prior_model.eval()
            residual_model.train(phase == 'train')

            residual_meter_loss = AverageMeter()
            residual_meter_loss_raw = AverageMeter()
            residual_meter_loss_ref = AverageMeter()
            residual_meter_loss_nxt = AverageMeter()
            residual_meter_loss_param = AverageMeter()

            for i, data in enumerate(tqdm(dataloaders[phase], desc=f'Epoch {residual_epoch}/{args.residual_n_epoch}')):
#                if i > 10:
#                    break
                if args.stage == 'dy':
                    # attrs: B x (n_p + n_s) x attr_dim
                    # particles: B x seq_length x (n_p + n_s) x state_dim
                    # n_particles: B
                    # n_shapes: B
                    # scene_params: B x param_dim
                    # Rrs, Rss: B x seq_length x n_rel x (n_p + n_s)
                    attrs, particles, n_particles, n_shapes, scene_params, Rrs, Rss, Rns, cluster_onehots = data
                    attrs = attrs.to(device)
                    particles = particles.to(device)
                    Rrs, Rss, Rns = Rrs.to(device), Rss.to(device), Rns.to(device)
                    if cluster_onehots is not None:
                        cluster_onehots = cluster_onehots.to(device)

                    # statistics
                    B = attrs.size(0)
                    n_particle = n_particles[0].item()
                    n_shape = n_shapes[0].item()

                    # p_rigid: B x n_instance
                    # p_instance: B x n_particle x n_instance
                    # physics_param: B x n_particle
                    groups_gt = get_env_group(args, n_particle, scene_params, use_gpu=use_gpu)

                    # memory: B x mem_nlayer x (n_particle + n_shape) x nf_memory
                    # for now, only used as a placeholder
                    memory_init = prior_model.module.init_memory(B, n_particle + n_shape)
                    loss = 0
                    pos_list = []
                    for j in range(args.sequence_length - args.n_his):
                        with torch.set_grad_enabled(phase == 'train'):
                            # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
                            if j == 0:
                                state_cur = particles[:, :args.n_his]
                                # Rrs_cur, Rss_cur: B x n_rel x (n_p + n_s)
                                Rr_cur = Rrs[:, args.n_his - 1]
                                Rs_cur = Rss[:, args.n_his - 1]
                                Rn_cur = Rns[:, args.n_his - 1]
                            else: # elif pred_pos.size(0) >= args.batch_size:
                                Rr_cur = []
                                Rs_cur = []
                                Rn_cur = []
                                max_n_rel = 0
                                for k in range(pred_pos.size(0)):
                                    _, _, Rr_cur_k, Rs_cur_k, Rn_cur_k, _ = prepare_input(pred_pos[k].detach().cpu().numpy(), n_particle, n_shape, args, stdreg=args.stdreg)
                                    Rr_cur.append(Rr_cur_k)
                                    Rs_cur.append(Rs_cur_k)
                                    Rn_cur.append(Rn_cur_k)
                                    max_n_rel = max(max_n_rel, Rr_cur_k.size(0))
                                for w in range(pred_pos.size(0)):
                                    Rr_cur_k, Rs_cur_k, Rn_cur_k = Rr_cur[w], Rs_cur[w], Rn_cur[w]
                                    Rr_cur_k = torch.cat([Rr_cur_k, torch.zeros(max_n_rel - Rr_cur_k.size(0), n_particle + n_shape)], 0)
                                    Rs_cur_k = torch.cat([Rs_cur_k, torch.zeros(max_n_rel - Rs_cur_k.size(0), n_particle + n_shape)], 0)
                                    Rn_cur_k = torch.cat([Rn_cur_k, torch.zeros(max_n_rel - Rn_cur_k.size(0), n_particle + n_shape)], 0)
                                    Rr_cur[w], Rs_cur[w], Rn_cur[w] = Rr_cur_k, Rs_cur_k, Rn_cur_k
                                Rr_cur = torch.from_numpy(np.stack(Rr_cur))
                                Rs_cur = torch.from_numpy(np.stack(Rs_cur))
                                Rn_cur = torch.from_numpy(np.stack(Rn_cur))
                                if use_gpu:
                                    Rr_cur = Rr_cur.to(device)
                                    Rs_cur = Rs_cur.to(device)
                                    Rn_cur = Rn_cur.to(device)
                                state_cur = torch.cat([state_cur[:,-3:], pred_pos.detach().unsqueeze(1)], dim=1)


                            if cluster_onehots is not None:
                                cluster_onehot = cluster_onehots[:, args.n_his - 1]
                            else:
                                cluster_onehot = None
                            # predict the velocity at the next time step
                            inputs = [attrs, state_cur, Rr_cur, Rs_cur, Rn_cur, memory_init, groups_gt, cluster_onehot]

                            # pred_pos (unnormalized): B x n_p x state_dim
                            # pred_motion_norm (normalized): B x n_p x state_dim
                            prior_pred_pos_p, _, _ = prior_model(inputs, j)
                            
                            gt_pos = particles[:, args.n_his + j]
                            gt_pos_p = gt_pos[:, :n_particle]
                            prior_pred_pos = torch.cat([prior_pred_pos_p, gt_pos[:, n_particle:]], 1).unsqueeze(1)
                            residual_inputs = [attrs, state_cur, Rr_cur, Rs_cur, Rn_cur, memory_init, groups_gt, cluster_onehot, prior_pred_pos]

                            # set_trace()
                            # print(torch.where(memory_init !=0))

                            pred_pos_p, pred_motion_norm, std_cluster = residual_model(residual_inputs, j)
                            # concatenate the state of the shapes
                            # pred_pos (unnormalized): B x (n_p + n_s) x state_dim
                            gt_pos = particles[:, args.n_his + j]
                            gt_pos_p = gt_pos[:, :n_particle]
                            # gt_sdf = sdf_list[:, args.n_his]
                            pred_pos = torch.cat([pred_pos_p, gt_pos[:, n_particle:]], 1)
                            
                            pos_list.append([pred_pos.detach().cpu().numpy(), gt_pos.detach().cpu().numpy()])

                            # gt_motion_norm (normalized): B x (n_p + n_s) x state_dim
                            # pred_motion_norm (normalized): B x (n_p + n_s) x state_dim
                            # gt_motion_norm should match then calculate if matched_motion enabled
                            if args.matched_motion:
                                gt_motion = matched_motion(particles[:, args.n_his], particles[:, args.n_his - 1], n_particles=n_particle)
                            else:
                                gt_motion = particles[:, args.n_his] - particles[:, args.n_his - 1]

                            mean_d, std_d = prior_model.module.stat[2:]
                            gt_motion_norm = (gt_motion - mean_d) / std_d
                            pred_motion_norm = torch.cat([pred_motion_norm, gt_motion_norm[:, n_particle:]], 1)
                            if args.loss_type == 'emd_chamfer_h':
                                if args.emd_weight > 0:
                                    emd_l = args.emd_weight * emd_loss(pred_pos_p, gt_pos_p)
                                    loss += emd_l
                                if args.chamfer_weight > 0:
                                    chamfer_l = args.chamfer_weight * chamfer_loss(pred_pos_p, gt_pos_p)
                                    loss += chamfer_l
                                if args.h_weight > 0:
                                    h_l = args.h_weight * h_loss(pred_pos_p, gt_pos_p)
                                    loss += h_l
                                # print(f"EMD: {emd_l.item()}; Chamfer: {chamfer_l.item()}; H: {h_l.item()}")
                            else:
                                raise NotImplementedError

                            if args.stdreg:
                                loss += args.stdreg_weight * std_cluster
                            loss_raw = F.l1_loss(pred_pos_p, gt_pos_p)

                            residual_meter_loss.update(loss.item(), B)
                            residual_meter_loss_raw.update(loss_raw.item(), B)
                    

                    # with open(args.outf + '/train.npy', 'wb') as f:
                    #     np.save(f, training_stats)
                if phase == "train":
                    residual_total_step += 1

                # update model parameters
                if phase == 'train':
                    residual_optimizer.zero_grad()
                    loss.backward()
                    residual_optimizer.step()
                
                torch.distributed.barrier()
                loss = reduce_mean(loss, num_gpus)
                emd_l = reduce_mean(emd_l, num_gpus)
                chamfer_l = reduce_mean(chamfer_l, num_gpus)

                if i % args.log_per_iter == 0:
                    print()
                    print('residual %s epoch[%d/%d] iter[%d/%d] LR: %.6f, loss: %.6f (%.6f), loss_raw: %.8f (%.8f)' % (
                        phase, residual_epoch, args.residual_n_epoch, i, len(dataloaders[phase]), get_lr(residual_optimizer),
                        loss.item(), residual_meter_loss.avg, loss_raw.item(), residual_meter_loss_raw.avg))
                    print('std_cluster', std_cluster)
                    if phase == 'train':
                        # torch.distributed.barrier()
                        residual_training_stats['loss'].append(loss.item())
                        residual_training_stats['loss_raw'].append(loss_raw.item())
                        residual_training_stats['iters'].append(residual_epoch * len(dataloaders[phase]) + i)
                
                if phase == "train":
                    if i % args.wandb_train_log_per_iter == 0 and dist.get_rank() == 0:  
                        wandb.log({f"{phase}_residual_total_weighted_loss" : loss.item()}) #, step=this_step)
                        wandb.log({f"{phase}_residual_emd_weighted_loss_1" : emd_l.item()}) #, step=this_step)
                        wandb.log({f"{phase}_residual_chamfer_weighted_loss_1" : chamfer_l.item()}) #, step=this_step)
                elif phase == "valid":
                    if i % args.wandb_valid_log_per_iter == 0 and dist.get_rank() == 0:
                        wandb.log({f"{phase}_residual_total_weighted_loss" : loss.item()}) #, step=this_step)
                        wandb.log({f"{phase}_residual_emd_weighted_loss_1" : emd_l.item()}) #, step=this_step)
                        wandb.log({f"{phase}_residual_chamfer_weighted_loss_1" : chamfer_l.item()}) #, step=this_step)
                
                if i % args.wandb_vis_log_per_iter == 0 and dist.get_rank() == 0:
                    for pstep_idx, pos in enumerate(pos_list):
                        pred_pos_np, gt_pos_np = pos
                        plt_render_image_split(pred_pos.detach().cpu().numpy(), gt_pos.detach().cpu().numpy(), n_particle, pstep_idx=pstep_idx)
                        for step in range(B):
                            wandb.log({f"{phase}_vis_plot_step_{str(pstep_idx)}": wandb.Image(f'visualize/step_{str(pstep_idx)}_bs_{str(step)}.png')})
                    
                

                if phase == 'train' and i > 0 and ((residual_epoch * len(dataloaders[phase])) + i) % args.ckp_per_iter == 0:
                    model_path = '%s/residual_net_epoch_%d_iter_%d' % (args.outf, residual_epoch, i)
                    # exists_or_mkdir(model_path)
                    if dist.get_rank() == 0:
                        exists_or_mkdir(model_path)
                        model_path = os.path.join(model_path, f"residual_model.pth")
                        save_checkpoint(epoch=residual_epoch, model=residual_model, optimizer=residual_optimizer, step=residual_total_step, save_path=model_path)
                    # torch.save(prior_model.state_dict(), model_path)
                    residual_rollout_epoch = residual_epoch
                    residual_rollout_iter = i


            print('residual %s epoch[%d/%d] Loss: %.6f, Best valid: %.6f' % (
                phase, residual_epoch, args.residual_n_epoch, residual_meter_loss.avg, residual_best_valid_loss))
            
            if dist.get_rank() == 0:
                with open(args.outf + '/residual_train.npy','wb') as f:
                    np.save(f, residual_training_stats)

            if phase == 'valid':
                torch.distributed.barrier()
                residual_meter_loss_avg_gather = distributed_concat(torch.from_numpy(np.array([residual_meter_loss.avg])).to(device))
                residual_meter_loss_avg_mean = np.mean(residual_meter_loss_avg_gather.detach().cpu().numpy().tolist())
                scheduler.step(residual_meter_loss_avg_mean)
                if residual_meter_loss_avg_mean < residual_best_valid_loss:
                    residual_best_valid_loss = residual_meter_loss_avg_mean
                    if dist.get_rank() == 0:
                        best_model_path = '%s/residual_net_best' % (args.outf)
                        exists_or_mkdir(best_model_path)
                        best_model_path = os.path.join(best_model_path, "best_residual_model.pth")
                        save_checkpoint(epoch=residual_epoch, model=residual_model, optimizer=residual_optimizer, step=residual_total_step, save_path=best_model_path)
    
    if dist.get_rank() == 0:             
        wandb.finish()
    pass



if __name__ == '__main__':
    args = gen_args()
    set_seed(args.random_seed)
    args.outf = os.path.join(args.outf, str(args.exp_id))
    exists_or_mkdir(args.dataf)
    exists_or_mkdir(args.outf)
    # os.system('mkdir -p ' + args.dataf)
    # os.system('mkdir -p ' + args.outf)

    tee = Tee(os.path.join(args.outf, 'train.log'), 'w')


    main(args)


