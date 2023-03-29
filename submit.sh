#!/bin/bash -l
# The above line must always be first, and must have "-l"
#SBATCH -J NodeDP
#SBATCH -p datasci
#SBATCH --output=results/logs/clean.out
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
module load python
conda activate torch

for RUN in 1 2 3 4 5
do
    python main.py --folds 5 --seed $RUN --batch_size 128 --epochs 400 --lr 0.05 --dataset cora --mode clean
done

for RUN in 1 2 3 4 5
do
    python main.py --folds 5 --seed $RUN --batch_size 128 --epochs 400 --lr 0.05 --dataset citeseer --mode clean
done

for RUN in 1 2 3 4 5
do
    python main.py --folds 5 --seed $RUN --batch_size 128 --epochs 400 --lr 0.05 --dataset pubmed --mode clean
done