import torch

def check_parallel(path1, path2):
    ckpt1 = torch.load(path1, map_location='cpu')["model_state_dict"]
    ckpt2 = torch.load(path2, map_location='cpu')["model_state_dict"]

    if ckpt1.keys() != ckpt2.keys():
        print("模型的状态字典键不匹配")
        return False
    # print(ckpt1.keys())
    # 检查每个参数和缓冲区是否相等
    for key in ckpt1.keys():
        if not torch.equal(ckpt1[key], ckpt2[key]):
            print("模型的参数或缓冲区不匹配")
            return False

if __name__ == "__main__":
    path1 = f"/nvme/tianyang/residual_robocake_data/dump_ngrip_fixed/residual_net_epoch_0_iter_5/residual_model_cuda:1.pth"
    path2 = f"/nvme/tianyang/residual_robocake_data/dump_ngrip_fixed/residual_net_epoch_0_iter_5/residual_model_cuda:2.pth"
    check_parallel(path1, path2)