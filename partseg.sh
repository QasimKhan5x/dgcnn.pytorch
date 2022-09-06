#! /bin/bash
MASTER_ADDR=localhost
MASTER_PORT=12355
NUM_TRAINERS=5
BATCH_SIZE=$( expr $NUM_TRAINERS \* 32 )
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS main_partseg_dist.py \
 --exp_name augmentations --batch_size $BATCH_SIZE  --k 32 \
 --use_height --use_sgd --epochs 500 \
 --decay 0.0001 --scheduler cycle --lr 0.000385