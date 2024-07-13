#!/bin/bash
#SBATCH -A m2651_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH -J GNN_tune_SchNet_4GPU
#SBATCH -o GNN_tune_SchNet_4GPU.log
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --mail-user=tommy_lin@berkeley.edu
#SBATCH --mail-type=ALL

export SLURM_CPU_BIND="cores"

ray start --head --num-cpus 4 --num-gpus 4
srun python main.py config_template_tune_SchNet.yml
ray stop