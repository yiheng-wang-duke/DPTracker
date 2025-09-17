# DPTracker
A simple and effective tracking framework for nighttime UAV Tracking.

## Deep learning environment
This repo is using Pytorch 2.1.2 with CUDA 11.8.

## Setup
Create the default local file with paths:
```bash
python tracking/create_default_local_file.py \
    --workspace_dir ./ \
    --data_dir ./data \
    --save_dir ./output
```
## Train
```bash train.sh```
## Test
```bash test.sh```
