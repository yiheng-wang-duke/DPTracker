export PYTHONPATH=/home/user/.conda/envs/wyh/lib/python3.8/site-packages:$PYTHONPATH
# python profile_model.py --script avtrack --config vit_tiny_patch16_224
python profile_model.py --script avtrack_pt --config vit_tiny_patch16_224
# python profile_model.py --script ostrack --config vitb_256_mae_ce_32x4_ep300
# python profile_model.py --script ostrack_pt --config vitb_256_ce_ep30
