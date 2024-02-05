#!/bin/bash
#SBATCH --comment=cifar100_10
#SBATCH --mem=88G
#SBATCH --account=dcs-acad4
#SBATCH --partition=dcs-acad
#SBATCH --time=6-0:0:0
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mail-user=xl.wang@sheffield.ac.uk

# Load the conda module
module load Anaconda3/5.3.0
# Load cuda
module load cuDNN/8.0.4.30-CUDA-11.1.1
# Activate conda
source activate SPG

pip install -r requirements.txt

cd /home/acq21xw/SPG

python main.py appr=gnr seq=femnist_10

