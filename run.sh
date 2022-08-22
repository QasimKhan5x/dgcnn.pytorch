#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=12355
NUM_TRAINERS=2
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS main_partseg_dist.py --exp_name svd_without_dgcnn --batch_size 24 --emb_dim 512 --ff_dims 512  --k 32 --n_heads 2 --n_blocks 2 --use_custom_attention