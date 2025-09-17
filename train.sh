export PYTHONPATH=/home/user/.conda/envs/wyh/lib/python3.8/site-packages:$PYTHONPATH
# CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/train.py --script dptrack_b --config vitb_256_ce_ep30 --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 0
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/train.py --script dptrack_t --config vit_tiny_patch16_224 --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 0

