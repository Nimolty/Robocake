CUDA_VISIBLE_DEVICES=1 python train_prior_desktop.py \
                    	--stage dy \
                    	--data_type ngrip_fixed \
                    	--prior_n_epoch 100 \
                    	--ckp_per_iter 100 \
                    	--eval 1 \
                    	--dataf /nvme/tianyang/prior_robocake_data/data_ngrip_fixed \
                    	--outf /nvme/tianyang/prior_robocake_data/dump_ngrip_fixed_desktop \
                    	--valid 0 \
                      --exp_id 1
	# --n_rollout 5
