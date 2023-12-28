# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 30211 /home/tianyang/Robocake/train_prior_distributed.py \
#   --stage dy \
# 	--data_type ngrip_fixed \
# 	--prior_n_epoch 5000 \
# 	--ckp_per_iter 500 \
#   --wandb_vis_log_per_iter 200 \
#   --dataf /nvme/tianyang/residual_robocake_data/data_ngrip_fixed \
#   --outf /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed_prior_multi \
#   --exp_id 0 \
#   --experiment_name prior_model_multi_v100v4 \
#   --batch_size 8 \
#   --num_worker 8 \
#   --resume_prior_path /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed_prior_multi/0/prior_net_epoch_3_iter_187/prior_model.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 30211 /home/tianyang/Robocake/train_prior_distributed.py \
  --stage dy \
	--data_type ngrip_fixed \
	--prior_n_epoch 5000 \
	--ckp_per_iter 1500 \
  --wandb_vis_log_per_iter 200 \
  --dataf /nvme/tianyang/residual_robocake_data/data_ngrip_fixed \
  --outf /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed_prior_multi \
  --exp_id 0 \
  --experiment_name prior_model_multi_v100v4_exp0 \
  --batch_size 8 \
  --num_worker 8 \
  --eval_ckp_per_iter 1 \
  --vis None \
  --resume_prior_path /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed_prior_multi/0/prior_net_epoch_51_iter_679/prior_model.pth

  CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 30212 /home/tianyang/Robocake/train_prior_distributed.py \
  --stage dy \
	--data_type ngrip_fixed \
	--prior_n_epoch 5000 \
	--ckp_per_iter 1500 \
  --wandb_vis_log_per_iter 200 \
  --dataf /nvme/tianyang/residual_robocake_data/data_ngrip_fixed \
  --outf /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed_prior_multi \
  --exp_id 2 \
  --experiment_name prior_model_multi_v100v4_exp2 \
  --batch_size 8 \
  --num_worker 8 \
  --eval_ckp_per_iter 1 \
  --vis None 
  