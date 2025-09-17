export PYTHONPATH=/home/user/.conda/envs/wyh/lib/python3.8/site-packages:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test.py dptrack_b vitb_256_ce_ep30 --dataset nat2021 --threads 4 --num_gpus 4 --debug 0
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test.py dptrack_b vitb_256_ce_ep30 --dataset uavdark135 --threads 4 --num_gpus 4 --debug 0
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test.py dptrack_b vitb_256_ce_ep30 --dataset darktrack2021 --threads 4 --num_gpus 4 --debug 0

CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test.py dptrack_t vit_tiny_patch16_224 --dataset nat2021 --threads 4 --num_gpus 4 --debug 0
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test.py dptrack_t vit_tiny_patch16_224 --dataset uavdark135 --threads 4 --num_gpus 4 --debug 0
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test.py dptrack_t vit_tiny_patch16_224 --dataset darktrack2021 --threads 4 --num_gpus 4 --debug 0