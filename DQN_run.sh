#!/bin/bash
#SBATCH --job-name=simplest
#SBATCH --partition=power-uriobols-gpu
#SBATCH --mem=64GB
#SBATCH --account=uriobols-users
#SBATCH --time=7-00:00:00
#SBATCH --output=/uriobolslab/yossi/my_storage/job_%j.txt
#SBATCH --error=/uriobolslab/yossi/my_storage/job_%j.err
##$SBATCH --cpus-per-task=1
##SBATCH --gres=gpu:1

echo "Running on node: $SLURM_JOB_NODELIST"
echo "CPUs allocated: $SLURM_JOB_CPUS_PER_NODE"

module load mamba/mamba-1.5.8

# Setting up conda environment directories
export CONDA_ENVS_DIRS=/tmp/python_cached_env
export CONDA_PKGS_DIRS=/tmp/python_cached_env

# Activate the conda environment using mamba
conda activate /a/home/cc/medicin/yossinahmias/my_storage/envs/epidemiology

# Change to the directory where the code is located
cd /a/home/cc/medicin/yossinahmias/my_storage/Epidemiology_DQN_Control/src
python _run.py
conda deactivate
