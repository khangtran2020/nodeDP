#!/bin/bash -l
# The above line must always be first, and must have "-l"
#SBATCH -J NodeDP
#SBATCH -p datasci
#SBATCH --output=results/logs/adult_ns_0.8.out
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
module load python
conda activate torch

for NS in 29.6 6.665 3.66 1.78 1.145 0.9295 0.815
do
    for RUN in 1 2 3 4 5
    do
        python main.py --folds 5 --seed $RUN --sampling_rate 0.05 --epochs 400 --lr 0.05 --clip 0.1 --ns $NS --clip_node 2 --dataset cora --mode dp --trim_rule random
    done
done