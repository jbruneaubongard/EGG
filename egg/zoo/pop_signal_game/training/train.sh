#!/bin/bash
#SBATCH -J small_test_training
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
python train.py --nb_agents=5 --save_data=True --path_save_exp_data=/homedtcl/jbruneaubongard/results
