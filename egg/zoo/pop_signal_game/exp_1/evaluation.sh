#!/bin/bash
#SBATCH -J eval_test
#SBATCH -p alien
#SBATCH -N 1
#SBATCH --mem 96G
#SBATCH --gres=gpu:1

#SBATCH -o slurm.%x.%J.%u.%N.out 
 
 
module load CUDA/11.4.3
module load Miniconda3/4.9.2
conda init bash
source ~/.bashrc
conda activate egg37
python evaluation.py --path_agents_data=/homedtcl/jbruneaubongard/results/2023-02-15_14h15_exp_1_fully_connected_nb-agents=5
