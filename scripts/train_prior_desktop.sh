CUDA_VISIBLE_DEVICES=0 python train_prior_desktop.py \
                    	--stage dy \
                    	--data_type ngrip_fixed \
                    	--prior_n_epoch 100 \
                    	--ckp_per_iter 100 \
                    	--eval 1 \
                    	--dataf /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/data/data_ngrip_fixed \
                    	--outf /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/data/dump_ngrip_fixed_desktop \
						--run_dir /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/wandb \
                    	--valid 1 \
                        --exp_id 1
	# --n_rollout 5
