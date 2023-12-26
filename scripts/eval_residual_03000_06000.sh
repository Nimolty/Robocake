# CUDA_VISIBLE_DEVICES=5 python eval_residual.py \
#                     	--stage dy \
#                     	--data_type ngrip_fixed \
#                     	--dataf /nvme/tianyang/residual_robocake_data/data_ngrip_fixed \
#                     	--outf /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed \
#                         --exp_id 0 \
#                         --resume_residual_path /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed/0/residual_net_epoch_6_iter_374/residual_model.pth \
# 						--resume_prior_path /home/tianyang/Robocake/prior/dump_ngrip_fixed_desktop/3/prior_net_epoch_99_iter_1325/prior_model.pth \
#                         --n_rollout 60 \
#                         --eval_data_class E_03000_06000

CUDA_VISIBLE_DEVICES=5 python eval_residual.py \
                    	--stage dy \
                    	--data_type ngrip_fixed \
                    	--dataf /nvme/tianyang/residual_robocake_data/data_ngrip_fixed \
                    	--outf /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed \
                        --exp_id 1 \
                        --resume_residual_path /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed/1/residual_net_epoch_6_iter_374/residual_model.pth \
						--resume_prior_path /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed_prior_multi/0/prior_net_epoch_6_iter_374/prior_model.pth \
                        --n_rollout 60 \
                        --eval_data_class E_03000_06000 \
						--vis None

CUDA_VISIBLE_DEVICES=2 python eval_residual.py \
                    	--stage dy \
                    	--data_type ngrip_fixed \
                    	--dataf /nvme/tianyang/residual_robocake_data/data_ngrip_fixed \
                    	--outf /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed \
                        --exp_id 2 \
                        --resume_residual_path /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed/2/residual_net_epoch_6_iter_374/residual_model.pth \
						--resume_prior_path /home/tianyang/Robocake/prior/dump_ngrip_fixed_desktop/3/prior_net_epoch_99_iter_1325/prior_model.pth \
                        --n_rollout 60 \
                        --eval_data_class E_03000_06000 \
						--vis None \
						--remove_his_particles 1


CUDA_VISIBLE_DEVICES=5 python eval_residual.py \
                    	--stage dy \
                    	--data_type ngrip_fixed \
                    	--dataf /nvme/tianyang/residual_robocake_data/data_ngrip_fixed \
                    	--outf /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed \
                        --exp_id 1 \
                        --resume_residual_path /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed/1/residual_net_epoch_15_iter_935/residual_model.pth \
						--resume_prior_path /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed_prior_multi/0/prior_net_epoch_6_iter_374/prior_model.pth \
                        --n_rollout 60 \
                        --eval_data_class E_03000_06000 \
						--vis None


CUDA_VISIBLE_DEVICES=2 python eval_residual.py \
                    	--stage dy \
                    	--data_type ngrip_fixed \
                    	--dataf /nvme/tianyang/residual_robocake_data/data_ngrip_fixed \
                    	--outf /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed \
                        --exp_id 2 \
                        --resume_residual_path /nvme/tianyang/residual_robocake_data/dump_ngrip_fixed/2/residual_net_epoch_15_iter_935/residual_model.pth \
						--resume_prior_path /home/tianyang/Robocake/prior/dump_ngrip_fixed_desktop/3/prior_net_epoch_99_iter_1325/prior_model.pth \
                        --n_rollout 60 \
                        --eval_data_class E_03000_06000 \
						--vis None \
						--remove_his_particles 1