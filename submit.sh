#!/bin/bash -l
# The above line must always be first, and must have "-l"
#SBATCH -J NodeDP
#SBATCH -p datasci
#SBATCH --output=results/logs/clean.out
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
module load python
conda activate torch


# mlp submission

python main.py --mode mlp \
        --mlp_mode clean \
        --dataset facebook \
        --model_type mlp \
        --lr 0.01 \
        --sampling_rate 0.01 \
        --n_layers 2 \
        --hid_dim 16 \
        --epochs 200 \
        --clip 1.0 \
        --ns 0.622 \
        --debug 0 \
        --n_neighbor 3 \
        --seed 1 \
        --dropout 0.0 \
        --device gpu
