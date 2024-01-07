

# CUDA_VISIBLE_DEVICES=0 python control_residual.py \
#                         --stage control \
#                         --data_type ngrip_fixed \
#                         --resume_prior_path  /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/dump/dump_ngrip_fixed_desktop/3/prior_net_epoch_99_iter_1325/prior_model.pth \
#                         --resume_residual_path /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/dump/dump_ngrip_fixed/0/residual_net_epoch_49_iter_1721/residual_model.pth \
#                         --dataf /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/data/data_ngrip_fixed \
#                         --shooting_size 200 \
#                         --control_algo predict \
#                         --n_grips 5 \
#                         --predict_horizon 2 \
#                         --opt_algo GD \
#                         --correction 1 \
#                         --shape_type alphabet \
#                         --goal_shape_name A \
#                         --debug 1 \
#                         --residual_input_next_action ZERO

#!/bin/bash

# 定义一个一维数组，其中每两个元素组成一个子列表
Shape_array=("A" "B" "D" "F" "G" "H" "I" "J" "K" "L" "M" "N" "O" "P" "Q" "R" "S" "T" "U" "V" "W" "X" "Y" )
#Shape_array=("G" "H" "I" "J" "K" "L" "M" "N" "O" "P" "Q" "R" "S" "T" "U" "V" "W" "X" "Y" )
#Shape_array=("E" "Z" "C" "Y")
E_array=(500.0 1000.0 1500.0 4000.0 5500.0 6000.0 8000.0 9000.0 9500.0)
nu_array=(0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2)
# E_array=(5500.0)
# nu_array=(0.2)
length=${#E_array[@]}
for ((j=0; j<length; j+=1)); do
    this_shape=${Shape_array[j]}
    for ((i=0; i<length; i+=1)); do
        this_E=${E_array[i]}
        this_nu=${nu_array[i]}

        echo "this_shape: $this_shape"
        echo "this_E: $this_E"
        echo "this_nu: $this_nu"

        CUDA_VISIBLE_DEVICES=0 python control_residual.py \
                        --stage control \
                        --data_type ngrip_fixed \
                        --resume_prior_path  /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/dump/dump_ngrip_fixed_desktop/3/prior_net_epoch_99_iter_1325/prior_model.pth \
                        --resume_residual_path /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/dump/dump_ngrip_fixed/3/residual_net_epoch_113_iter_877/residual_model.pth \
                        --dataf /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/data/data_ngrip_fixed \
                        --shooting_size 200 \
                        --control_algo predict \
                        --n_grips 5 \
                        --predict_horizon 2 \
                        --opt_algo CEM \
                        --correction 1 \
                        --shape_type alphabet \
                        --goal_shape_name $this_shape \
                        --debug 0 \
                        --residual_input_next_action ZERO \
                        --sc_material_E $this_E \
                        --sc_material_nu $this_nu


        CUDA_VISIBLE_DEVICES=0 python control_prior.py \
                        --stage control \
                        --data_type ngrip_fixed \
                        --resume_prior_path  /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/dump/dump_ngrip_fixed_prior_multi/2/prior_net_epoch_113_iter_877/prior_model.pth \
                        --dataf /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/data/data_ngrip_fixed \
                        --shooting_size 200 \
                        --control_algo predict \
                        --n_grips 5 \
                        --predict_horizon 2 \
                        --opt_algo CEM \
                        --correction 1 \
                        --shape_type alphabet \
                        --goal_shape_name $this_shape \
                        --debug 0 \
                        --sc_material_E $this_E \
                        --sc_material_nu $this_nu


        CUDA_VISIBLE_DEVICES=0 python control_prior.py \
                        --stage control \
                        --data_type ngrip_fixed \
                        --resume_prior_path  /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/dump/dump_ngrip_fixed_desktop/3/prior_net_epoch_99_iter_1325/prior_model.pth \
                        --dataf /home/nimolty/Nimolty_Research/Basic_Settings/Robocake/data/data_ngrip_fixed \
                        --shooting_size 200 \
                        --control_algo predict \
                        --n_grips 5 \
                        --predict_horizon 2 \
                        --opt_algo CEM \
                        --correction 1 \
                        --shape_type alphabet \
                        --goal_shape_name $this_shape \
                        --debug 0 \
                        --sc_material_E $this_E \
                        --sc_material_nu $this_nu
        
    done
done
