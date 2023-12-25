CUDA_VISIBLE_DEVICES=7 python eval_residual.py \
                    	--stage dy \
                    	--data_type ngrip_fixed \
                    	--dataf /nvme/tianyang/residual_robocake_data/data_ngrip_fixed \
                    	--outf /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed \
                        --exp_id 0 \
                        --resume_residual_path /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed/0/residual_net_epoch_6_iter_374/residual_model.pth \
						--resume_prior_path /home/tianyang/Robocake/prior/dump_ngrip_fixed_desktop/3/prior_net_epoch_99_iter_1325/prior_model.pth \
                        --n_rollout 60