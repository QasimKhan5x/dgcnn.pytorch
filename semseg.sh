#! /bin/bash
MASTER_ADDR=localhost
MASTER_PORT=12355
NUM_TRAINERS=1
BATCH_SIZE=$( expr $NUM_TRAINERS \* 8 )
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS main_semseg_dist.py \
 --exp_name exp --batch_size $BATCH_SIZE  --k 16 --epochs 100 \
 --use_height --use_sgd --test_batch_size $NUM_TRAINERS \
 --decay 0.0001 --scheduler cycle --lr 0.000385