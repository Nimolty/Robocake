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
from datasets.dataset import DoughDataset, TestDoughDataset
from utils.robocraft_utils import prepare_input, get_scene_info, get_env_group
from metrics.metric import ChamferLoss, EarthMoverLoss, HausdorffLoss
from utils.optim import get_lr, count_parameters, my_collate, AverageMeter, Tee, get_optimizer, distributed_concat
from utils.utils import set_seed, matched_motion, load_checkpoint, save_checkpoint, exists_or_mkdir, load_model, reduce_mean
from visualize.visualize import plt_render_image_split
from pdb import set_trace

### load model ###
from models.prior_model_distributed import Prior_Model

### parallel training ###
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist

###   socket   ###
import wandb
import socket

### load eval functions ###
from eval_prior_multiprocessing import prepare_model_and_data, inference_after_training, print_eval



def main(args):
    ########################## set local rank ##########################
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device=torch.device("cuda",args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        use_gpu = True
    else:
        raise NotImplementedError
    num_gpus = torch.cuda.device_count()

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
        drop_last= phase=="train", 
        collate_fn=my_collate) for phase in phases} # TODO: understand the logics of my_collate
    

    for eval_data_class in args.eval_data_class_list:
        test_root_dir = os.path.join(args.dataf, eval_data_class)
        this_test_dataset = TestDoughDataset(test_root_dir, args)
        datasets[eval_data_class] = this_test_dataset
        samplers[eval_data_class] = DistributedSampler(datasets[eval_data_class])
        dataloaders[eval_data_class] = DataLoader(
        datasets[eval_data_class],
        sampler=samplers[eval_data_class],
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True
        )

    ########################## create model ##########################
    prior_model = Prior_Model(args, device).to(device)
    print("model #params: %d" % count_parameters(prior_model))
    prior_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(prior_model).to(device)
    prior_model = torch.nn.parallel.DistributedDataParallel(prior_model, device_ids=[args.local_rank],
                                                output_device=args.local_rank,find_unused_parameters=True)


    ########################## load pretrained model ##########################
    if args.resume_prior_path:
        print("Loading saved prior ckpt from %s" % args.resume_prior_path)

        if args.stage == 'dy':
            prior_checkpoint = load_checkpoint(args.resume_prior_path, device)
            prior_model = load_model(prior_model, prior_checkpoint['model_state_dict'])
    

    ########################## create optimizer ##########################
    if args.stage == 'dy':
        prior_params = prior_model.parameters()
    else:
        raise AssertionError("unknown stage: %s" % args.stage)
    
    prior_optimizer = get_optimizer(params=prior_params, optimizer_mode=args.optimizer, lr=args.lr, beta1=args.beta1)
    # reduce learning rate when a metric has stopped improving
    scheduler = ReduceLROnPlateau(prior_optimizer, 'min', factor=0.8, patience=3, verbose=True)

    if args.resume_prior_path:
        if args.stage == 'dy':
            prior_optimizer.load_state_dict(prior_checkpoint['optimizer_state_dict'])
    
    ########################## define loss ##########################
    chamfer_loss = ChamferLoss()
    emd_loss = EarthMoverLoss()
    h_loss = HausdorffLoss()


    ########################## start training ##########################
    prior_start_epoch = 0
    if args.resume_prior_path:
        if args.stage == 'dy':
            prior_start_epoch = prior_checkpoint['epoch']

    prior_best_valid_loss = np.inf
    prior_training_stats = {'args':vars(args), 'loss':[], 'loss_raw':[], 'iters': [], 'loss_emd': [], 'loss_motion': []}
    prior_rollout_epoch = -1
    prior_rollout_iter = -1
    prior_total_step = 0
    prior_valid_step = 0
    if args.resume_prior_path:
        if args.stage == 'dy':
            prior_total_step = prior_checkpoint["epoch"] * (int(datasets["train"].__len__()) / args.batch_size / num_gpus)  
            prior_valid_step = prior_checkpoint["epoch"] * (int(datasets["valid"].__len__()) / args.batch_size / num_gpus) 

    for prior_epoch in range(prior_start_epoch, args.prior_n_epoch):
        for phase in phases:
            samplers[phase].set_epoch(prior_epoch)
            prior_model.train(phase == 'train')

            prior_meter_loss = AverageMeter()
            prior_meter_loss_raw = AverageMeter()
            prior_meter_loss_ref = AverageMeter()
            prior_meter_loss_nxt = AverageMeter()
            prior_meter_loss_param = AverageMeter()

            for i, data in enumerate(tqdm(dataloaders[phase], desc=f'Epoch {prior_epoch}/{args.prior_n_epoch}')):

                # if i > 10:
                #     break
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
                    loss_dict = {}
                    pos_list = []
                    # if i % args.vis_per_iter == 0:
                        
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
                            pred_pos_p, pred_motion_norm, std_cluster = prior_model(inputs, j, args.prior_remove_his_particles)

                            # concatenate the state of the shapes
                            # pred_pos (unnormalized): B x (n_p + n_s) x state_dim
                            gt_pos = particles[:, args.n_his + j]
                            gt_pos_p = gt_pos[:, :n_particle]
                            # gt_sdf = sdf_list[:, args.n_his]
                            pred_pos = torch.cat([pred_pos_p, gt_pos[:, n_particle:]], 1)
                            
                            pos_list.append([pred_pos.detach().cpu().numpy(), gt_pos.detach().cpu().numpy()])
                            
#                            if i % args.vis_per_iter == 0:
#                                print("render")
#                                plt_render_image_split(pred_pos.detach().cpu().numpy(), gt_pos.detach().cpu().numpy(), n_particle, pstep_idx=j)


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

                            prior_meter_loss.update(loss.item(), B)
                            prior_meter_loss_raw.update(loss_raw.item(), B)
                    
                # update model parameters
                if phase == 'train':
                    prior_optimizer.zero_grad()
                    loss.backward()
                    prior_optimizer.step()
                    # with open(args.outf + '/train.npy', 'wb') as f:
                    #     np.save(f, training_stats)

                if phase == "train":
                    prior_total_step += 1
                    this_step = prior_total_step
                elif phase == "valid":
                    prior_valid_step += 1
                    this_step = prior_valid_step
                    
                torch.distributed.barrier()
                loss = reduce_mean(loss, num_gpus)
                emd_l = reduce_mean(emd_l, num_gpus)
                chamfer_l = reduce_mean(chamfer_l, num_gpus)
                    
                if i % args.log_per_iter == 0:
                    print()
                    print('Prior %s epoch[%d/%d] iter[%d/%d] LR: %.6f, loss: %.6f (%.6f), loss_raw: %.8f (%.8f)' % (
                        phase, prior_epoch, args.prior_n_epoch, i, len(dataloaders[phase]), get_lr(prior_optimizer),
                        loss.item(), prior_meter_loss.avg, loss_raw.item(), prior_meter_loss_raw.avg))
                    print('std_cluster', std_cluster)
                    if phase == 'train':
                        prior_training_stats['loss'].append(loss.item())
                        prior_training_stats['loss_raw'].append(loss_raw.item())
                        prior_training_stats['iters'].append(prior_epoch * len(dataloaders[phase]) + i)
                
                # ### save wandb stats ###
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
                        plt_render_image_split(pred_pos.detach().cpu().numpy(), gt_pos.detach().cpu().numpy(), n_particle, pstep_idx=pstep_idx, vis_dir=args.vis_dir)
                        for step in range(B): 
                            wandb.log({f"{phase}_vis_plot_step_{str(pstep_idx)}": wandb.Image(f'{args.vis_dir}/step_{str(pstep_idx)}_bs_{str(step)}.png')})      

                if phase == 'train' and i > 0 and ((prior_epoch * len(dataloaders[phase])) + i) % args.ckp_per_iter == 0:
                    model_path = '%s/prior_net_epoch_%d_iter_%d' % (args.outf, prior_epoch, i)
                    if dist.get_rank() == 0:
                        exists_or_mkdir(model_path)
                        this_model_path = os.path.join(model_path, "prior_model.pth")
                        save_checkpoint(epoch=prior_epoch, model=prior_model, optimizer=prior_optimizer, step=prior_total_step, save_path=this_model_path)
                    # torch.save(prior_model.state_dict(), model_path)
                    args.resume_prior_path =  os.path.join(model_path, f"prior_model.pth")
                    prior_rollout_epoch = prior_epoch
                    prior_rollout_iter = i
                    if args.eval_ckp_per_iter:
                        for eval_data_class in args.eval_data_class_list:
                            args.eval_data_class = eval_data_class
                            loss_list_over_episodes = []
                            for test_i, data_dict in enumerate(tqdm(dataloaders[eval_data_class])):
                                prior_model, _, prior_eval_out_path = prepare_model_and_data(args, device, use_gpu, 
                                                                                        prior_model=prior_model)
                                loss_list = inference_after_training(prior_model, args, data_dict, prior_eval_out_path, use_gpu, device)
                                loss_list_over_episodes.append(loss_list)
                            
                            torch.distributed.barrier()
                            loss_list_over_episodes_gather = distributed_concat(torch.from_numpy(np.array(loss_list_over_episodes)).to(device))
                            loss_list_over_episodes = loss_list_over_episodes_gather.detach().cpu().numpy().tolist()
                            result_dict = print_eval(args, prior_eval_out_path, loss_list_over_episodes)
                            if dist.get_rank() == 0:
                                wandb.log({f"{eval_data_class} Last Frame EMD" : result_dict["Last Frame EMD"]})
                                wandb.log({f"{eval_data_class} Last Frame CD" : result_dict["Last Frame CD"]})
                                wandb.log({f"{eval_data_class} Last Frame HD" : result_dict["Last Frame HD"]})
                                wandb.log({f"{eval_data_class} Over Episodes EMD" : result_dict["Over Episodes EMD"]})
                                wandb.log({f"{eval_data_class} Over Episodes CD" : result_dict["Over Episodes CD"]})
                                wandb.log({f"{eval_data_class} Over Episodes HD" : result_dict["Over Episodes HD"]})

                        prior_model.train(phase=="train")


            print('Prior %s epoch[%d/%d] Loss: %.6f, Best valid: %.6f' % (
                phase, prior_epoch, args.prior_n_epoch, prior_meter_loss.avg, prior_best_valid_loss))
            
            if dist.get_rank() == 0:
                with open(args.outf + '/prior_train.npy','wb') as f:
                    np.save(f, prior_training_stats)

            # set_trace()
            if phase == 'valid':
                # set_trace()
                torch.distributed.barrier()
                prior_meter_loss_avg_gather = distributed_concat(torch.from_numpy(np.array([prior_meter_loss.avg])).to(device))
                prior_meter_loss_avg_mean = np.mean(prior_meter_loss_avg_gather.detach().cpu().numpy().tolist())
                scheduler.step(prior_meter_loss_avg_mean)
                if prior_meter_loss_avg_mean < prior_best_valid_loss:
                    prior_best_valid_loss = prior_meter_loss_avg_mean
                    if dist.get_rank() == 0:
                        best_model_path = '%s/prior_net_best' % (args.outf)
                        exists_or_mkdir(best_model_path)
                        best_model_path = os.path.join(best_model_path, "best_prior_model.pth")
                        save_checkpoint(epoch=prior_epoch, model=prior_model, optimizer=prior_optimizer, step=prior_total_step, save_path=best_model_path)

    if dist.get_rank() == 0:                    
        wandb.finish()
    pass



if __name__ == '__main__':
    args = gen_args()
    set_seed(args.random_seed)
    args.outf = os.path.join(args.outf, str(args.exp_id))
    args.vis_dir = os.path.join(args.outf, "visualize")
    exists_or_mkdir(args.dataf)
    exists_or_mkdir(args.outf)
    exists_or_mkdir(args.vis_dir)

    tee = Tee(os.path.join(args.outf, 'train.log'), 'w')


    main(args)


