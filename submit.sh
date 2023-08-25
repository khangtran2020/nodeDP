#!/bin/bash -l
# The above line must always be first, and must have "-l"
#SBATCH -J NodeDP
#SBATCH -p datasci
#SBATCH --output=results/logs/reddit_mlp.out
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
module load python
conda activate torch


# mlp submission

for RUN in 1 2 3 4 5
do 
	python main.py --mode mlp \
		--mlp_mode clean \
		--dataset reddit \
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
		--seed $RUN \
		--dropout 0.0 \
		--device gpu
done

for RUN in 1 2 3 4 5
do 
	python main.py --mode mlp \
		--mlp_mode dp \
		--dataset reddit \
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
		--seed $RUN \
		--dropout 0.0 \
		--device gpu
done
