

CUDA_VISIBLE_DEVICES=0 python eval_prior.py \
                    	--stage dy \
                    	--data_type ngrip_fixed \
                    	--dataf /nvme/tianyang/residual_robocake_data/data_ngrip_fixed \
                    	--outf /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed_prior_multi \
                        --exp_id 0 \
						--resume_prior_path /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed_prior_multi/0/prior_net_epoch_6_iter_374/prior_model.pth \
                        --n_rollout 60