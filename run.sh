#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=12355
NUM_TRAINERS=
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS main_partseg.py --arg1