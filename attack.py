import os
import sys
import datetime
import warnings
import torch
from Attacks.config import parse_args_attack
from Attacks.Data.read_attack import read_data
from Attacks.Utils.utils import print_args, seed_everything, generate_name, read_pickel, init_history
from Models.init import init_model, init_optimizer
from loguru import logger
from rich import print as rprint
from Attacks.node_attack import run as run_node

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
warnings.filterwarnings("ignore")



def run(args, current_time, device):

    if args.att_mode == 'node':
        run_node(args=args, current_date=current_time, device=device)



if __name__ == "__main__":

    current_time = datetime.datetime.now()
    args = parse_args_attack()
    print_args(args=args)
    args.debug = True if args.debug == 1 else False
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device == 'cpu':
        device = torch.device('cpu')
    rprint(f"DEVICE USING: {device}")
    run(args=args, current_time=current_time, device=device)
