# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 30210 /home/tianyang/Robocake/train_residual_distributed.py \
#   --stage dy \
# 	--data_type ngrip_fixed \
# 	--residual_n_epoch 5000 \
# 	--ckp_per_iter 500 \
#   --wandb_vis_log_per_iter 200 \
#   --dataf /nvme/tianyang/residual_robocake_data/data_ngrip_fixed \
#   --outf /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed \
#   --resume_prior_path /home/tianyang/Robocake/prior/dump_ngrip_fixed_desktop/3/prior_net_epoch_99_iter_1325/prior_model.pth \
#   --exp_id 0 \
#   --experiment_name residual_model_v100v4 \
#   --batch_size 8 \
#   --num_worker 8

########################## on going ##########################
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 30210 /home/tianyang/Robocake/train_residual_distributed.py \
  --stage dy \
	--data_type ngrip_fixed \
	--residual_n_epoch 5000 \
	--ckp_per_iter 500 \
  --wandb_vis_log_per_iter 200 \
  --dataf /nvme/tianyang/residual_robocake_data/data_ngrip_fixed \
  --outf /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed \
  --resume_prior_path /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed_prior_multi/0/prior_net_epoch_6_iter_374/prior_model.pth \
  --exp_id 1 \
  --experiment_name residual_model_v100v3 \
  --batch_size 8 \
  --num_worker 8

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 30211 /home/tianyang/Robocake/train_residual_distributed.py \
  --stage dy \
	--data_type ngrip_fixed \
	--residual_n_epoch 5000 \
	--ckp_per_iter 500 \
  --wandb_vis_log_per_iter 200 \
  --dataf /nvme/tianyang/residual_robocake_data/data_ngrip_fixed \
  --outf /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed \
  --resume_prior_path /home/tianyang/Robocake/prior/dump_ngrip_fixed_desktop/3/prior_net_epoch_99_iter_1325/prior_model.pth \
  --exp_id 2 \
  --experiment_name residual_model_v100v3 \
  --batch_size 8 \
  --num_worker 8 \
  --remove_his_particles 1