import datetime
import warnings
import sys
from config import parse_args
from Data.read import read_data, init_loader
from Models.init import init_model, init_optimizer
from Runs.run_clean import run as run_clean
from Runs.run_nodedp import run as run_nodedp
from Runs.run_mlp import run as run_mlp
from Runs.run_grad_inspect import run as run_grad_inspect
from Utils.utils import *
from loguru import logger
from rich import print as rprint

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
warnings.filterwarnings("ignore")


def run(args, current_time, device):

    if args.mode == 'clean':
        history = init_history_clean()
    else:
        history = init_history_nodeDP()
    save_args_to_history(args=args, history=history)

    with timeit(logger, 'init-data'):
        train_g, val_g, test_g, _ = read_data(args=args, data_name=args.dataset, history=history)
        print(f"Done get spectral for dataset {args.dataset} with spectral dens {args.density}")


if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args()
    print_args(args=args)
    args.debug = True if args.debug == 1 else False
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device == 'cpu':
        device = torch.device('cpu')
    rprint(f"DEVICE USING: {device}")
    run(args=args, current_time=current_time, device=device)
