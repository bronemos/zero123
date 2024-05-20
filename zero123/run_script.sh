source .venv/bin/activate

srun python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 main.py -t --base configs/sd-objaverse-finetune-c_concat-256-test.yaml --gpus 0,1,2,3,4,5,6,7 --scale_lr False --num_nodes 1 --seed 42 --check_val_every_n_epoch 10 --finetune_from /scratch/project_462000478/spieglb/checkpoints/105000.ckpt