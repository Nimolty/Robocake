import os

import numpy as np 
import cv2 
import json
import glob

def stat_cloth(visual_cloth_path):
    eval_dict = {"success": 1.0, "fail": 0.0}
    vis_cloth_path_list = glob.glob(os.path.join(visual_cloth_path, '*.jpg'))
    total_length = len(vis_cloth_path_list)
    
    score = 0.0
    success_num = 0
    for vis_cloth_path in vis_cloth_path_list:
        vis_name = vis_cloth_path.split('/')[-1].split('-')
        this_vis_score = float(vis_name[4])
        success_or_fail = eval_dict[vis_name[5].replace('.jpg', '')]
        score += this_vis_score / total_length
        success_num += success_or_fail
        
    print(f"param_list", visual_cloth_path)
    print(f"total_length: {total_length}")
    print(f"success_num: {success_num}")
    print(f"average_score: {score}")




if __name__ == "__main__":
    visual_cloth_path = f"./DeformableAffordance/visual/cloth"

    param_list = ["strech_0.5_bend_0.5_shear_0.5",
                  "strech_0.8_bend_1.0_shear_0.9",
                  "strech_1.5_bend_1.5_shear_1.5",
                  "strech_2.0_bend_0.5_shear_1.0",
                  "strech_1.5_bend_1.2_shear_1.5"
                ]
    for param in param_list:
        visual_cloth_path = f"./DeformableAffordance/visual/{param}/cloth"
        stat_cloth(visual_cloth_path)
