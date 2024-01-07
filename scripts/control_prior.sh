# CUDA_VISIBLE_DEVICES=0 python control_prior.py \
#                         --stage control \
#                         --data_type ngrip_fixed \
#                         --resume_prior_path  /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/dump/dump_ngrip_fixed_prior_multi/0/prior_net_epoch_50_iter_950/prior_model.pth \
#                         --dataf /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/data/data_ngrip_fixed \
#                         --shooting_size 200 \
#                         --control_algo predict \
#                         --n_grips 5 \
#                         --predict_horizon 2 \
#                         --opt_algo GD \
#                         --correction 1 \
#                         --shape_type alphabet \
#                         --goal_shape_name A \
#                         --debug 1


# CUDA_VISIBLE_DEVICES=0 python control_prior.py \
#                         --stage control \
#                         --data_type ngrip_fixed \
#                         --resume_prior_path  /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/dump/dump_ngrip_fixed_desktop/3/prior_net_epoch_99_iter_1325/prior_model.pth \
#                         --dataf /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/data/data_ngrip_fixed \
#                         --shooting_size 200 \
#                         --control_algo predict \
#                         --n_grips 5 \
#                         --predict_horizon 2 \
#                         --opt_algo GD \
#                         --correction 1 \
#                         --shape_type alphabet \
#                         --goal_shape_name C \
#                         --debug 1


# CUDA_VISIBLE_DEVICES=0 python control.py \
# 	--stage control \
#     --data_type ngrip_fixed \
# 	--dataf /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/data/data_ngrip_fixed \
# 	--resume_prior_path  /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/dump/out_25-Jun-2022-14:53:02.222126/net_epoch_19_iter_34.pth \
# 	--shooting_size 200 \
# 	--control_algo predict \
# 	--n_grips 5 \
# 	--predict_horizon 2 \
# 	--opt_algo GD \
# 	--correction 1 \
# 	--shape_type alphabet \
# 	--goal_shape_name B \
# 	--debug 1


# CUDA_VISIBLE_DEVICES=0 python control_prior.py \
#                         --stage control \
#                         --data_type ngrip_fixed \
#                         --resume_prior_path  /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/dump/out_25-Jun-2022-14:53:02.222126/net_epoch_19_iter_34.pth \
#                         --dataf /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/data/data_ngrip_fixed \
#                         --shooting_size 200 \
#                         --control_algo predict \
#                         --n_grips 5 \
#                         --predict_horizon 2 \
#                         --opt_algo GD \
#                         --correction 1 \
#                         --shape_type alphabet \
#                         --goal_shape_name A \
#                         --debug 1


