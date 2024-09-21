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
--checkpoint_dir=logs/vgg16_0920 \
--check_distance_value=0.01 \
--share_every=20 \
--save_every=5 \
--start_share_epoch=1 \
--share_height_type=whole \
--macro_height=64 \
--macro_width=64 \
--flow=row \
--no_share_initial \
--train_batch_size=128 \
--val_batch_size=320 \
--min_sharing_rate_per_macro=0.1 \
--train_subset_size=200 \
--reduced_val \
--conv_ratio=0.0 \
--fc_ratio=0.01 \
--dist_weight=10 \
--soft_weight=10.0 \
--pred_weight=0.0 \
--conv_ratio_list 0 0 0 0 0 0 0 .1 .1 .1 .1 .1 0 \
--fc_ratio_list 0 0 0 \
")
# --dist_type=l3norm \
# --train_subset_size=20000 \
