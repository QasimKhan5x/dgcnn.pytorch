#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=12355
NUM_TRAINERS=5
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS main_partseg_dist.py --emb_dim 512 --ff_dims 512 --batch_size 10 --k 20 --n_heads 2 --n_blocks 1 --use_custom_attention