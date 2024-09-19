import os

#os.system("CUDA_VISIBLE_DEVICES=2 python3 main_train_all.py --qkv_weight=1 --pred_weight=0.0 --soft_weight=1 --hidden_weight=1 --dist_weight=10 --loss_type=dist  --lr=1e-4   --epoch 60 --checkpoint_dir=logs/0707_min0_width16_qkv0.5 --no_grad_share --check_distance=0.49 --qkv_ratio=0.5 --fc1_ratio=0.0 --fc2_ratio=0.0  --share_every=5 --save_every=5 --routing_group=1 --macro_width=16  --with_dist --start_share_epoch=30 --macro_height=64 --train_subset_size=20000 --flow=row --no_share_initial --share_height_type=whole --reduced_val --train_batch_size=64 --min_sharing_rate_per_macro=0")
#os.system("CUDA_VISIBLE_DEVICES=1 python3 main_train_all.py --qkv_weight=1 --pred_weight=0.0 --soft_weight=1 --hidden_weight=1 --dist_weight=10 --loss_type=dist  --lr=1e-4   --epoch 60 --checkpoint_dir=logs/0707_min0.75_width16_qkvfc10.5       --load_checkpoint=logs/0707_min0.75_width16_qkv0.5/checkpoint_60.pt --no_grad_share --check_distance=0.49 --qkv_ratio=0.5 --fc1_ratio=0.5 --fc2_ratio=0.0  --share_every=5 --save_every=5 --routing_group=1 --macro_width=16  --with_dist --start_share_epoch=30 --macro_height=64 --train_subset_size=20000 --flow=row --no_share_initial --share_height_type=whole --reduced_val --train_batch_size=64 --min_sharing_rate_per_macro=0.75")
# os.system("CUDA_VISIBLE_DEVICES=3 python3 main_train_all.py --qkv_weight=1 --pred_weight=0.0 --soft_weight=1 --hidden_weight=1 --dist_weight=10 --loss_type=dist  --lr=5e-5   --epoch 5  --checkpoint_dir=logs/0707_min0.8_width16_qkvfc10.5_cont  --load_checkpoint=logs/0707_min0.8_width16_qkvfc10.5/checkpoint_60.pt --no_grad_share --check_distance=0.49 --qkv_ratio=0.5 --fc1_ratio=0.5 --fc2_ratio=0.0  --share_every=1 --save_every=1 --routing_group=1 --macro_width=16  --with_dist --start_share_epoch=1 --macro_height=64 --train_subset_size=-1 --flow=row --no_share_initial --share_height_type=whole --train_batch_size=64 --min_sharing_rate_per_macro=0.8 | tee teelogs/0707_min0.8_width16_qkvfc10.5_cont.txt")
# os.system("CUDA_VISIBLE_DEVICES=3 python3 main_train_all.py --qkv_weight=1 --pred_weight=0.0 --soft_weight=1 --hidden_weight=1 --dist_weight=10 --loss_type=dist  --lr=5e-5   --epoch 5  --checkpoint_dir=logs/0707_min0.8_width16_qkvfc10.5fc20.3 --load_checkpoint=logs/0707_min0.8_width16_qkvfc10.5_cont/checkpoint_5.pt --no_grad_share --check_distance=0.49 --qkv_ratio=0.5 --fc1_ratio=0.5 --fc2_ratio=0.3  --share_every=1 --save_every=1 --routing_group=1 --macro_width=16  --with_dist --start_share_epoch=1 --macro_height=64 --train_subset_size=-1 --flow=row --no_share_initial --share_height_type=whole --train_batch_size=64 --min_sharing_rate_per_macro=0.8 | tee teelogs/0707_min0.8_width16_qkvfc10.5fc20.3.txt")
# os.system("CUDA_VISIBLE_DEVICES=3 python3 main_train_all.py --qkv_weight=1 --pred_weight=0.0 --soft_weight=1 --hidden_weight=1 --dist_weight=10 --loss_type=dist  --lr=5e-5   --epoch 5  --checkpoint_dir=logs/0707_min0.8_width16_qkvfc10.5fc20.5 --load_checkpoint=logs/0707_min0.8_width16_qkvfc10.5fc20.3/checkpoint_5.pt --no_grad_share --check_distance=0.49 --qkv_ratio=0.5 --fc1_ratio=0.5 --fc2_ratio=0.5  --share_every=1 --save_every=1 --routing_group=1 --macro_width=16  --with_dist --start_share_epoch=1 --macro_height=64 --train_subset_size=-1 --flow=row --no_share_initial --share_height_type=whole --train_batch_size=64 --min_sharing_rate_per_macro=0.8 | tee teelogs/0707_min0.8_width16_qkvfc10.5fc20.5.txt")


#os.system("CUDA_VISIBLE_DEVICES=0 python3 main_train_all.py --qkv_weight=1 --pred_weight=0.0 --soft_weight=1 --hidden_weight=1 --dist_weight=10 --loss_type=dist  --lr=1e-5   --epoch 5  --checkpoint_dir=logs/0707_min0.8lr1e5_width16_qkvfc10.5_cont  --load_checkpoint=logs/0707_min0.8_width16_qkvfc10.5/checkpoint_60.pt --no_grad_share --check_distance=0.49 --qkv_ratio=0.5 --fc1_ratio=0.5 --fc2_ratio=0.0  --share_every=1 --save_every=1 --routing_group=1 --macro_width=16  --with_dist --start_share_epoch=1 --macro_height=64 --train_subset_size=-1 --flow=row --no_share_initial --share_height_type=whole --train_batch_size=64 --min_sharing_rate_per_macro=0.8 | tee teelogs/0707_min0.8lr1e5_width16_qkvfc10.5_cont.txt")
#os.system("CUDA_VISIBLE_DEVICES=0 python3 main_train_all.py --qkv_weight=1 --pred_weight=0.0 --soft_weight=1 --hidden_weight=1 --dist_weight=10 --loss_type=dist  --lr=1e-5   --epoch 5  --checkpoint_dir=logs/0707_min0.8lr1e5_width16_qkvfc10.5fc20.3 --load_checkpoint=logs/0707_min0.8lr1e5_width16_qkvfc10.5_cont/checkpoint_5.pt --no_grad_share --check_distance=0.49 --qkv_ratio=0.5 --fc1_ratio=0.5 --fc2_ratio=0.3  --share_every=1 --save_every=1 --routing_group=1 --macro_width=16  --with_dist --start_share_epoch=1 --macro_height=64 --train_subset_size=-1 --flow=row --no_share_initial --share_height_type=whole --train_batch_size=64 --min_sharing_rate_per_macro=0.8 | tee teelogs/0707_min0.8lr1e5_width16_qkvfc10.5fc20.3.txt")
#os.system("CUDA_VISIBLE_DEVICES=0 python3 main_train_all.py --qkv_weight=1 --pred_weight=0.0 --soft_weight=1 --hidden_weight=1 --dist_weight=10 --loss_type=dist  --lr=1e-5   --epoch 5  --checkpoint_dir=logs/0707_min0.8lr1e5_width16_qkvfc10.5fc20.5 --load_checkpoint=logs/0707_min0.8lr1e5_width16_qkvfc10.5fc20.3/checkpoint_5.pt --no_grad_share --check_distance=0.49 --qkv_ratio=0.5 --fc1_ratio=0.5 --fc2_ratio=0.5  --share_every=1 --save_every=1 --routing_group=1 --macro_width=16  --with_dist --start_share_epoch=1 --macro_height=64 --train_subset_size=-1 --flow=row --no_share_initial --share_height_type=whole --train_batch_size=64 --min_sharing_rate_per_macro=0.8 | tee teelogs/0707_min0.8lr1e5_width16_qkvfc10.5fc20.5.txt")

#CUDA_VISIBLE_DEVICES=0 python3 main_train_all.py --qkv_weight=1 --pred_weight=0.0 --soft_weight=1 --hidden_weight=1 --dist_weight=10 --loss_type=dist  --lr=5e-5   --epoch 5  --checkpoint_dir=logs/0719_min0.8_width16_qkv0.5fc10.3 --load_checkpoint=logs/0707_min0.8_width16_qkv0.5/checkpoint_60.pt --no_grad_share --check_distance_value=0.01 --qkv_ratio=0.5 --fc1_ratio=0.3 --fc2_ratio=0.0  --share_every=1 --save_every=1 --routing_group=1 --macro_width=16  --with_dist --start_share_epoch=1 --macro_height=64 --train_subset_size=-1 --flow=row --no_share_initial --share_height_type=whole --train_batch_size=64 --min_sharing_rate_per_macro=0.8 | tee teelogs/0719_min0.8_width16_qkvfc10.3.txt

#CUDA_VISIBLE_DEVICES=2 python3 main_train_all.py --qkv_weight=1 --pred_weight=0.0 --soft_weight=1 --hidden_weight=1 --dist_weight=10 --loss_type=dist  --lr=1e-4   --epoch 60 --checkpoint_dir=logs/temp --check_distance=0.49 --qkv_ratio=0.5 --fc1_ratio=0.0 --fc2_ratio=0.0  --share_every=5 --save_every=5 --macro_width=16 --start_share_epoch=30 --macro_height=64 --train_subset_size=20000 --flow=row --no_share_initial --share_height_type=whole --reduced_val --train_batch_size=64 --min_sharing_rate_per_macro=0

#os.system("CUDA_VISIBLE_DEVICES=3 python3 main_train_all.py --qkv_weight=1 --pred_weight=0.0 --soft_weight=1 --hidden_weight=1 --dist_weight=10 --loss_type=dist  --lr=1e-4   --epoch 60 --checkpoint_dir=logs/0731_min0.8_width16_blockratio0.8_qkv0.5 --check_distance_value=0.01 --qkv_ratio=0.5 --fc1_ratio=0.0 --fc2_ratio=0.0  --share_every=5 --save_every=5 --macro_width=16 --start_share_epoch=30 --macro_height=64 --train_subset_size=20000 --flow=row --no_share_initial --share_height_type=whole --reduced_val --train_batch_size=64 --min_sharing_rate_per_macro=0.8 --block_ratio=0.8")

os.system("CUDA_VISIBLE_DEVICES=3 python3 main_train_all.py \
--qkv_weight=1 \
--pred_weight=0.0 \
--soft_weight=1 \
--hidden_weight=1 \
--dist_weight=10 \
--loss_type=dist \
--lr=7e-5 \
--min_lr=1e-5 \
--epoch=60 \
--checkpoint_dir=logs/0904_min1_width16_blockratio1_qkv0.5 \
--check_distance_value=0.01 \
--qkv_ratio=0.5 \
--fc1_ratio=0.0 \
--fc2_ratio=0.0 \
--max_qkv_ratio=0.5 \
--max_fc1_ratio=0.0 \
--max_fc2_ratio=0.0 \
--share_every=5 \
--save_every=5 \
--macro_width=16 \
--start_share_epoch=30 \
--macro_height=64 \
--train_subset_size=20000 \
--flow=row \
--no_share_initial \
--share_height_type=whole \
--train_batch_size=64 \
--min_sharing_rate_per_macro=1.0 \
--block_ratio=1 \
--max_ratio_epoch=4 \
")

os.system("CUDA_VISIBLE_DEVICES=3 python3 main_train_all.py \
--qkv_weight=1 \
--pred_weight=0.0 \
--soft_weight=1 \
--hidden_weight=1 \
--dist_weight=10 \
--loss_type=dist \
--lr=7e-5 \
--min_lr=1e-5 \
--epoch=5 \
--load_checkpoint=logs/0904_min1_width16_blockratio1_qkv0.5/checkpoint_60.pt \
--checkpoint_dir=logs/0904_min1_width16_blockratio1_qkvfc10.5 \
--check_distance_value=0.01 \
--qkv_ratio=0.5 \
--fc1_ratio=0.5 \
--fc2_ratio=0.0 \
--max_qkv_ratio=0.5 \
--max_fc1_ratio=0.5 \
--max_fc2_ratio=0.0 \
--share_every=1 \
--save_every=1 \
--macro_width=16 \
--start_share_epoch=0 \
--macro_height=64 \
--train_subset_size=-1 \
--flow=row \
--no_share_initial \
--share_height_type=whole \
--train_batch_size=64 \
--min_sharing_rate_per_macro=1.0 \
--block_ratio=1 \
--max_ratio_epoch=4 \
")

os.system("CUDA_VISIBLE_DEVICES=3 python3 main_train_all.py \
--qkv_weight=1 \
--pred_weight=0.0 \
--soft_weight=1 \
--hidden_weight=1 \
--dist_weight=10 \
--loss_type=dist \
--lr=7e-5 \
--min_lr=1e-5 \
--epoch=5 \
--load_checkpoint=logs/0904_min1_width16_blockratio1_qkvfc10.5/checkpoint_best.pt \
--checkpoint_dir=logs/0904_min1_width16_blockratio1_qkvfc10.5fc20.3 \
--check_distance_value=0.01 \
--qkv_ratio=0.5 \
--fc1_ratio=0.5 \
--fc2_ratio=0.3 \
--max_qkv_ratio=0.5 \
--max_fc1_ratio=0.5 \
--max_fc2_ratio=0.3 \
--share_every=1 \
--save_every=1 \
--macro_width=16 \
--start_share_epoch=0 \
--macro_height=64 \
--train_subset_size=-1 \
--flow=row \
--no_share_initial \
--share_height_type=whole \
--train_batch_size=64 \
--min_sharing_rate_per_macro=1.0 \
--block_ratio=1 \
--max_ratio_epoch=4 \
")

# os.system("CUDA_VISIBLE_DEVICES=3 python3 main_train_all.py \
# --qkv_weight=1 \
# --pred_weight=0.0 \
# --soft_weight=1 \
# --hidden_weight=1 \
# --dist_weight=100 \
# --loss_type=dist \
# --lr=1 \
# --epoch 1 \
# --checkpoint_dir=logs/temp0813 \
# --check_distance_value=0.01 \
# --qkv_ratio=0.05 \
# --fc1_ratio=0.05 \
# --fc2_ratio=0.03 \
# --max_qkv_ratio=0.5 \
# --max_fc1_ratio=0.5 \
# --max_fc2_ratio=0.3 \
# --share_every=1 \
# --save_every=1 \
# --macro_width=16 \
# --start_share_epoch=0 \
# --macro_height=64 \
# --train_subset_size=12800 \
# --flow=row \
# --no_share_initial \
# --share_height_type=whole \
# --train_batch_size=64 \
# --min_sharing_rate_per_macro=0 \
# --block_ratio=1 \
# --best_acc=0.8 \
# --ratio_change_step=10 \
# --max_ratio_epoch=1 \
# ")

# os.system("CUDA_VISIBLE_DEVICES=3 python3 main_train_all.py --qkv_weight=1 --pred_weight=0.0 --soft_weight=1 --hidden_weight=1 --dist_weight=10 --loss_type=dist  --lr=5e-5   --epoch 5  --checkpoint_dir=logs/0803_min0.8_width16_blockratio0.8_qkvfc10.5_cont  --load_checkpoint=logs/0803_min0.8_width16_blockratio0.8_qkvfc10.5/checkpoint_60.pt --check_distance_value=0.01 --qkv_ratio=0.5 --fc1_ratio=0.5 --fc2_ratio=0.0  --share_every=1 --save_every=1 --macro_width=16  --start_share_epoch=1 --macro_height=64 --train_subset_size=-1 --flow=row --no_share_initial --share_height_type=whole --train_batch_size=64 --min_sharing_rate_per_macro=0.8 --block_ratio=0.8")
# os.system("CUDA_VISIBLE_DEVICES=3 python3 main_train_all.py --qkv_weight=1 --pred_weight=0.0 --soft_weight=1 --hidden_weight=1 --dist_weight=10 --loss_type=dist  --lr=5e-5   --epoch 5  --checkpoint_dir=logs/0803_min0.8_width16_blockratio0.8_qkvfc10.5fc20.3 --load_checkpoint=logs/0803_min0.8_width16_blockratio0.8_qkvfc10.5_cont/checkpoint_best.pt --check_distance_value=0.01 --qkv_ratio=0.5 --fc1_ratio=0.5 --fc2_ratio=0.3  --share_every=1 --save_every=1  --macro_width=16 --start_share_epoch=1 --macro_height=64 --train_subset_size=-1 --flow=row --no_share_initial --share_height_type=whole --train_batch_size=64 --min_sharing_rate_per_macro=0.8 --block_ratio=0.8")
# os.system("CUDA_VISIBLE_DEVICES=3 python3 main_train_all.py --qkv_weight=1 --pred_weight=0.0 --soft_weight=1 --hidden_weight=1 --dist_weight=10 --loss_type=dist  --lr=5e-5   --epoch 5  --checkpoint_dir=logs/0803_min0.8_width16_blockratio0.8_qkvfc10.5fc20.5 --load_checkpoint=logs/0803_min0.8_width16_blockratio0.8_qkvfc10.5fc20.3/checkpoint_best.pt --check_distance_value=0.01 --qkv_ratio=0.5 --fc1_ratio=0.5 --fc2_ratio=0.5  --share_every=1 --save_every=1 --macro_width=16 --start_share_epoch=1 --macro_height=64 --train_subset_size=-1 --flow=row --no_share_initial --share_height_type=whole --train_batch_size=64 --min_sharing_rate_per_macro=0.8 --block_ratio=0.8")
