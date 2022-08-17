#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=12355
NUM_TRAINERS=2
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS main_partseg_dist.py --batch_size 8 --k 29 --n_heads 1 --n_blocks 2