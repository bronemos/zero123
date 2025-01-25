#!/bin/bash

source .venv/bin/activate

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export TORCH_NCCL_USE_COMM_NONBLOCKING=1
# export TORCH_NCCL_NONBLOCKING_TIMEOUT=1800
python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 main.py -t --base configs/sd-objaverse-finetune-c_concat-256-test.yaml --gpus 0,1,2,3,4,5,6,7 --scale_lr False --num_nodes 1 --seed 42 --check_val_every_n_epoch 10 --finetune_from /scratch/project_462000478/spieglb/checkpoints/105000.ckpt