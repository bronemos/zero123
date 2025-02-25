#!/bin/bash

module use /appl/local/csc/modulefiles/
module load pytorch

export SING_FLAGS="--overlay /scratch/project_462000478/spieglb/datasets/objaverse/objaverse_0.sqfs --overlay /scratch/project_462000478/spieglb/datasets/objaverse/objaverse_1.sqfs --overlay /scratch/project_462000478/spieglb/datasets/objaverse/objaverse_2.sqfs --overlay /scratch/project_462000478/spieglb/datasets/objaverse/objaverse_3.sqfs --overlay /scratch/project_462000478/spieglb/datasets/objaverse/objaverse_4.sqfs --overlay /scratch/project_462000478/spieglb/datasets/objaverse/objaverse_5.sqfs --overlay /scratch/project_462000478/spieglb/datasets/objaverse/objaverse_6.sqfs --overlay /scratch/project_462000478/spieglb/datasets/objaverse/objaverse_7.sqfs --overlay /scratch/project_462000478/spieglb/datasets/objaverse/objaverse_8.sqfs --overlay /scratch/project_462000478/spieglb/datasets/objaverse/objaverse_9.sqfs -B /scratch/project_462000478 $SING_FLAGS"

singularity_wrapper shell

source .venv/bin/activate

