#!/bin/bash
#SBATCH -A m2651_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 6:00:00
#SBATCH -J GNN_tune_SchNet
#SBATCH -o GNN_tune_SchNet.log
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --mail-user=tommy_lin@berkeley.edu
#SBATCH --mail-type=ALL

export SLURM_CPU_BIND="cores"
# srun python main.py config_train_MEGNet_gap.yml
# srun python main.py config_train_SchNet_gap.yml

# ray start --head --num-gpus 1
# srun python main.py config_tune_SchNet_gap.yml
# ray stop

# nohup python main.py config_tune_SchNet_gap.yml > GNN_tune_SchNet_manual.log &

# nohup python main.py config_inference_SchNet_gap.yml > GNN_inference_SchNet.log &