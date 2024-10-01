#!/bin/bash

#SBATCH --job-name=SC
#SBATCH --output=/home/c01chzh/CISPA-projects/pt_network-2024/tmp/job-%j.out
#SBATCH --error=/home/c01chzh/CISPA-projects/pt_network-2024/tmp/slurm-%j.err

#SBATCH --partition=r65257773x
#SBATCH --time=6-23:30:00

if [ ! -f ~/.config/enroot/.credentials ]; then
        mkdir -p ~/.config/enroot/
        ln -s ~/CISPA-home/.config/enroot/.credentials ~/.config/enroot/.credentials
fi

JOBDATADIR=$HOME/CISPA-projects/pt_network-2024/"$SLURM_JOB_ID"
JOBTMPDIR=$HOME/CISPA-projects/pt_network-2024/tmp/job-"$SLURM_JOB_ID"

srun mkdir -p "$JOBDATADIR" "$JOBTMPDIR"

srun --container-image=projects.cispa.saarland:5005#c01chzh/docker-test:v4 \
        --container-mounts="$JOBTMPDIR":/tmp \
        python3 $HOME/CISPA-projects/pt_network-2024/dictionary_learning_from_scratch/main_log.py
srun mv $HOME/CISPA-projects/pt_network-2024/tmp/job-"$SLURM_JOB_ID".out "$JOBDATADIR"/out.txt
# srun mv "$JOBTMPDIR"/ "$JOBDATADIR"/slurm
# srun -p gpu --container-image=projects.cispa.saarland:5005#c01chzh/docker-test:v1 --time=01:00:00 --pty bash
