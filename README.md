# DPTracker

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)

A simple and effective tracking framework for nighttime UAV Tracking.

<p align="center">
  <img src="assets/overview.png" alt="DPTracker Overview" width="800"/>
</p>

## 📣 News
- **[2026.03]** Model checkpoints and raw tracking results are now available!
- **[2026.02]** Our paper has been accepted by ICRA 2026! 🎉

## 📁 Download
- **Model Checkpoints**:
  - [Baidu Netdisk](https://pan.baidu.com/s/1Dr7cq8hMi6qK3u006OQ51Q) (Code: ac7z)
  - [Google Drive](https://drive.google.com/drive/folders/1YkUyK5V7F63DMDP0W8FZPAXLCXWrT2Ex?usp=drive_link)
- **Raw Tracking Results**:
  - [Baidu Netdisk](https://pan.baidu.com/s/1CJLEbk9XfFOSstZllTrINg) (Code: haax)
  - [Google Drive](https://drive.google.com/drive/folders/1wxpvaYglka2jMWmR2sG0qVJKtjuhnyjX?usp=drive_link)

## 🖥️ Environment
This repo is using Pytorch 2.1.2 with CUDA 11.8.

## 🛠️ Setup
Create the default local file with paths:
```bash
python tracking/create_default_local_file.py \
    --workspace_dir ./ \
    --data_dir ./data \
    --save_dir ./output
```

## 📈 Train
### Preparation
Download the training data, including GOT-10K, LASOT, COCO, and TrackingNet, ExDark, Shift, and BDD100K.

### Run
```
conda activate your_env
bash train.sh
```

## 📊 Test
### Preparation 
Download the test data, including NAT2021, UAVDark135, DarkTrack2021.

### Run
```
conda activate your_env
bash test.sh
```
