import os
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type = str, 
        help = 'yaml config file')    
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument('--num_workers', type=int, default=4)   
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')
    parser.add_argument('--resume_dvae_ckpt', type=str, default="")
    # parser.add_argument("")
    parser.add_argument("--dataf", type=str, default="")
    parser.add_argument("--outf", type=str, default="")
    parser.add_argument("--valid", type=int, default=1)
    parser.add_argument("--team_name", type=str, default="nimolty-and-his-codes")

    ### experiment logging ###
    parser.add_argument("--run_dir", type=str, default="/nvme/tianyang/wandb")
    parser.add_argument("--exp_id", type=int, default=0)
    parser.add_argument("--project_name", type=str, default="Robocake")
    parser.add_argument("--experiment_name", type=str,default="dvae_pretrain")
    parser.add_argument("--scenario_name", type=str, default="nimolty")
    parser.add_argument("--wandb_vis_log_per_iter", type=int, default=500)
    parser.add_argument("--wandb_train_log_per_iter", type=int, default=500)
    parser.add_argument("--wandb_valid_log_per_iter", type=int, default=200)

    ### training ###
    parser.add_argument("--batch_size", type=int, default=4)


    
    
    args = parser.parse_args()
    return args


