#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=12355
NUM_TRAINERS=4
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS main_partseg_dist.py --exp_name enc2dec1_pointnet_no_pos --batch_size 64 --emb_dims 512 --ff_dims 512  --k 32 --n_heads 2 --n_blocks 2