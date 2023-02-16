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
python exp3.py --nb_agents=5 --subtype_exp=noncentral-noncentral --save_data=True --path_save_exp_data=/homedtcl/jbruneaubongard/results --w_central_sender=2 --w_central_receiver=2 --w_noncentral=1 --w_bridge_sender=1 --w_bridge_receiver=1 --path_agents_data=/homedtcl/jbruneaubongard/results/2023-02-13_16h32_training_nb-agents=5
 