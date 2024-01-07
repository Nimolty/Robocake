import numpy as np
import glob
import os
from tqdm import tqdm

# def transfer(from_path, to_path):
#     from_path_list = glob.glob(os.path.join(from_path, "0*"))
#     to_path_list = glob.glob(os.path.join(to_path, "0*"))
#     from_path_list.sort()
#     to_path_list.sort()
#     print(len(from_path_list))
#     print(len(to_path_list))

#     for from_path, to_path in tqdm(zip(from_path_list, to_path_list)):
#         assert from_path.split("/")[-1] == to_path.split("/")[-1]
#         path_idx = from_path.split("/")[-1]
        
#         for j in range(120):
#             gtp = np.load(from_path + f"/{j:03d}_gtp.npy", allow_pickle=True)
#             with open(f"{to_path}/{j:03d}_gtp.npy", 'wb') as f:
#                 np.save(f, gtp)
            
#             d_and_p = np.load(from_path + f"/{j:03d}_depth_prim.npy", allow_pickle=True)
#             primitives = d_and_p[4:6]
#             with open(f"{to_path}/{j:03d}_depth_prim.npy", 'wb') as f:
#                 np.save(f, primitives)

def transfer(from_path, to_path):
    from_path_list = glob.glob(os.path.join(from_path, "0063", "*"))
    to_path_list = glob.glob(os.path.join(to_path, "0063", "*"))
    from_path_list.sort()
    to_path_list.sort()
    print(len(from_path_list))
    print(len(to_path_list))

    for from_path, to_path in tqdm(zip(from_path_list, to_path_list)):
        assert from_path.split("/")[-1] == to_path.split("/")[-1]
        assert from_path.split("/")[-2] == to_path.split("/")[-2]
        # path_idx = from_path.split("/")[-1]
        
        for j in range(120):
            gtp = np.load(from_path + f"/{j:03d}_gtp.npy", allow_pickle=True)
            with open(f"{to_path}/{j:03d}_gtp.npy", 'wb') as f:
                np.save(f, gtp)
            
            d_and_p = np.load(from_path + f"/{j:03d}_depth_prim.npy", allow_pickle=True)
            primitives = d_and_p[4:6]
            with open(f"{to_path}/{j:03d}_depth_prim.npy", 'wb') as f:
                np.save(f, primitives)



if __name__ == "__main__":
    from_path = f"/home/nimolty/Nimolty_Research/Basic_Settings/Robocake_data_0107/dataset/ngrip_fixedtrain_07-Jan-2024-00:25:25.158616"
    to_path = f"/home/nimolty/Nimolty_Research/Basic_Settings/Robocake_data_0107/dataset/sample_output"
    transfer(from_path, to_path)