# CUDA_VISIBLE_DEVICES=0 python main_dvae.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 30211 main_dvae.py \
        --dataf /nvme/tianyang/sample_pybert_data \
        --outf /nvme/tianyang/sample_pybert_data/test \
        --config /home/tianyang/Robocake/pcbert/cfgs/ShapeNet55_models/dvae.yaml