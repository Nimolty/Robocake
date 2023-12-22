CUDA_VISIBLE_DEVICES=2,3,4 python -m torch.distributed.launch --nproc_per_node=3 --master_port 30210 /home/tianyang/Robocake/train_residual_distributed.py \
    --stage dy \
	--data_type ngrip_fixed \
	--residual_n_epoch 5000 \
	--ckp_per_iter 100 \
	--eval 1 \
    --dataf /nvme/tianyang/residual_robocake_data/data_ngrip_fixed \
    --outf /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed