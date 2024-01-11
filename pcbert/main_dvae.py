import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

from pdb import set_trace
# set_trace()

### load preliminary ###
import os
import numpy as np
import math

### import torch ###
import torch
import torch.nn.functional as F
from torch.autograd import Variable

### parallel training ###
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
from torch.utils.data import DataLoader

### load datasets ###
from pcbert.pcdatasets.dvae_dataset import ShapeNet

### load configs ###
from pcbert.pcutils.config import *
from pcbert.pcutils import misc
from pcbert.pcutils.misc import worker_init_fn
from pcbert.pcutils.parser import get_args
from pcbert.pcutils.metrics import Metrics
from pcbert.pcutils.utils import set_seed, exists_or_mkdir, load_checkpoint, save_checkpoint, load_model, load_info
from pcbert.pcutils.utils import build_opti_sche, reduce_mean
from pcbert.pcutils.AverageMeter import AverageMeter
### wandb ###
import wandb 
import socket

### load models ###
from pcbert.models.dvae import DiscreteVAE

### loss ###
from pcbert.extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

from tqdm import tqdm
# def simple_main():
#     pass

#     config_path = f"/home/tianyang/Robocake/pcbert/cfgs/ShapeNet55_models/dvae.yaml"
#     config = cfg_from_yaml_file(config_path)
#     # set_trace()
#     model = DiscreteVAE(config["model"])
#     model  = model.cuda()

#     for i in range(100):
#         points = torch.from_numpy(np.random.random((16, 300, 3))).float().cuda()
#         output = model(points)

#         set_trace()
def compute_loss(loss_1, loss_2, config, niter):
    '''
    compute the final loss for optimization
    For dVAE: loss_1 : reconstruction loss, loss_2 : kld loss
    '''
    start = config.kldweight.start
    target = config.kldweight.target
    ntime = config.kldweight.ntime

    _niter = niter - 10000
    if _niter > ntime:
        kld_weight = target
    elif _niter < 0:
        kld_weight = 0.
    else:
        kld_weight = target + (start - target) *  (1. + math.cos(math.pi * float(_niter) / ntime)) / 2.

    loss = loss_1 + kld_weight * loss_2

    return loss


def get_temp(config, niter):
    if config.get('temp') is not None:
        start = config.temp.start
        target = config.temp.target
        ntime = config.temp.ntime
        if niter > ntime:
            return target
        else:
            temp = target + (start - target) *  (1. + math.cos(math.pi * float(niter) / ntime)) / 2.
            return temp
    else:
        return 0 


def main(args):
    config = cfg_from_yaml_file(args.config)

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device=torch.device("cuda",args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        use_gpu = True
    else:
        raise NotImplementedError
    num_gpus = torch.cuda.device_count()


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
    
    phases = ['train'] if args.valid == 0 else ['train', 'valid']
    datasets = {phase: ShapeNet(args, phase) for phase in phases}
    samplers = {phase: DistributedSampler(datasets[phase]) for phase in phases}

    print(f"Train dataset size: {len(datasets['train'])}")
    dataloaders = {phase: DataLoader(
        datasets[phase],
        sampler=samplers[phase],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last= phase=="train", 
        worker_init_fn=worker_init_fn) for phase in phases}

    model = DiscreteVAE(config["model"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                output_device=args.local_rank,find_unused_parameters=True)

    start_epoch = 0
    best_metrics = None
    metrics = None

    if args.resume_dvae_ckpt:
        checkpoint = load_checkpoint(args.resume_dvae_ckpt, device)
        model = load_model(model, checkpoint['model_state_dict'])
        start_epoch, best_metrics = load_info(checkpoint)
        best_metrics = Metrics(config.consider_metric, best_metrics)

    optimizer, scheduler = build_opti_sche(model, config)
    if args.resume_dvae_ckpt:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        for phase in phases:
            samplers[phase].set_epoch(epoch)
            model.train(phase == 'train')

            # epoch_start_time = time.time()
            # batch_start_time = time.time()
            # batch_time = AverageMeter()
            # data_time = AverageMeter()
            losses = AverageMeter(['Loss1', 'Loss2'])

            num_iter = 0
            n_batches = len(dataloaders["train"])
            for idx, data in enumerate(tqdm(dataloaders[phase], desc=f'Epoch {epoch}/{config.max_epoch}')):
                if phase == "train":
                    num_iter += 1
                    n_itr = epoch * n_batches + idx
                    temp = get_temp(config, n_itr)
                    points = data.float().to(device)
                    ret = model(points, temperature = temp, hard = False)

                    loss_1, loss_2 = model.module.get_loss(ret, points)

                    _loss = compute_loss(loss_1, loss_2, config, n_itr)

                    _loss.backward()

                    if num_iter == config.step_per_update:
                        num_iter = 0
                        optimizer.step()
                        model.zero_grad()
                    
                    torch.distributed.barrier()
                    loss_1 = reduce_mean(loss_1, num_gpus)
                    loss_2 = reduce_mean(loss_2, num_gpus)
                    losses.update([loss_1.item() * 1000, loss_2.item() * 1000])

                    torch.cuda.synchronize()

                    if n_itr % args.wandb_train_log_per_iter == 0 and dist.get_rank() == 0:  
                        wandb.log({f"{phase}_loss_1" : loss_1.item()}) #, step=this_step)
                        wandb.log({f"{phase}_loss_2" : loss_2.item()})
                        wandb.log({f'{phase}_Loss/Batch/Temperature' : temp}, step=n_itr)
                        wandb.log({f'{phase}_Loss/Batch/LR' : optimizer.param_groups[0]['lr'] }, step=n_itr)
                elif phase == "valid":
                    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
                    test_metrics = AverageMeter(Metrics.names())
                    n_samples = len(dataloaders[phase])
                    with torch.no_grad():
                        points = data.float().to(device)
                        ret = model(inp = points, hard=True, eval=True)
                        coarse_points = ret[0]
                        dense_points = ret[1]

                        sparse_loss_l1 =  ChamferDisL1(coarse_points, points)
                        sparse_loss_l2 =  ChamferDisL2(coarse_points, points)
                        dense_loss_l1 =  ChamferDisL1(dense_points, points)
                        dense_loss_l2 =  ChamferDisL2(dense_points, points)

                        torch.distributed.barrier()
                        sparse_loss_l1 = reduce_mean(sparse_loss_l1, num_gpus)
                        sparse_loss_l2 = reduce_mean(sparse_loss_l2, num_gpus)
                        dense_loss_l1 = reduce_mean(dense_loss_l1, num_gpus)
                        dense_loss_l2 = reduce_mean(dense_loss_l2, num_gpus)

                        test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])



            if config.scheduler.type != 'function' and phase == "train":
                if isinstance(scheduler, list):
                    for item in scheduler:
                        item.step(epoch)
                else:
                    scheduler.step(epoch)
            
            if phase == "train" and dist.get_rank() == 0:
                wandb.log({f'{phase}_Loss/Epoch/Loss_1': losses.avg(0)})
                wandb.log({f'{phase}_Loss/Epoch/Loss_2': losses.avg(1)})
                model_path = os.path.join(args.outf, str(epoch).zfill(4))
                exists_or_mkdir(model_path)
                save_checkpoint(epoch, model, optimizer, n_itr, metrics, best_metrics, 
                            save_path=os.path.join(model_path, "model.pth"))
            elif phase == "valid" and dist.get_rank() == 0:
                wandb.log({f'{phase}_Loss/Epoch/Sparse': test_losses.avg(0)})
                wandb.log({f'{phase}_Loss/Epoch/Dense': test_losses.avg(2)})



                        

    




    





if __name__ == "__main__":
    args = get_args()
    set_seed(args.random_seed)
    args.outf = os.path.join(args.outf, str(args.exp_id))
    args.vis_dir = os.path.join(args.outf, "visualize")
    exists_or_mkdir(args.dataf)
    exists_or_mkdir(args.outf)
    exists_or_mkdir(args.vis_dir)

    main(args)

