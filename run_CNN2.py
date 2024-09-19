import os
import argparse
import torch

parser = argparse.ArgumentParser(description="Train CNN with specific CPU core")
parser.add_argument('cpu', type=int, default=0, help='Index of the CPU core to use')
args = parser.parse_args()

os.system(f"CUDA_VISIBLE_DEVICES={args.cpu} python3 main_train_all_CNN.py \
--loss_type=dist \
--lr=1e-4 \
--min_lr=1e-5 \
--epoch=100 \
--checkpoint_dir=logs/test_CNN2 \
--check_distance_value=0.01 \
--share_every=5 \
--save_every=5 \
--start_share_epoch=1 \
--macro_width=64 \
--macro_height=64 \
--flow=row \
--no_share_initial \
--share_height_type=whole \
--train_batch_size=64 \
--val_batch_size=128 \
--min_sharing_rate_per_macro=0.1 \
--train_subset_size=100 \
--conv_ratio=0.5 \
--fc_ratio=0.0 \
--reduced_val \
")
# --reduced_val \
# --dist_type=l3norm \
# --train_subset_size=20000 \
