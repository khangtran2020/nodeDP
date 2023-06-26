import datetime
import warnings
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch
from config import parse_args
from Data.read import read_data, init_loader
from Utils.utils import seed_everything, get_name, timeit
from Models.init import init_model, init_optimizer
from Runs.run_clean import run as run_clean
from Runs.run_nodedp import run as run_nodedp
from Runs.run_dp import run as run_dp
from Utils.utils import print_args
from loguru import logger
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
warnings.filterwarnings("ignore")


def run(args, current_time, device):

    fold = 0
    with timeit(logger, 'init-data'):
        train_g, val_g, test_g = read_data(args=args, data_name=args.dataset)
        tr_loader, va_loader, te_loader = init_loader(args=args, device=device, train_g=train_g, test_g=test_g, val_g=val_g)

    model = init_model(args=args)
    optimizer = init_optimizer(optimizer_name=args.optimizer, model=model, lr=args.lr)
    name = get_name(args=args, current_date=current_time, fold=fold)

    tr_info = (train_g, tr_loader)
    va_info = va_loader
    te_info = (test_g, te_loader)

    run_dict = {
        'clean': run_clean,
        'nodedp': run_nodedp,
        'dp': run_dp
    }
    run_mode = run_dict[args.mode]

    run_mode(args=args, tr_info=tr_info, va_info=va_info, te_info=te_info, model=model,
             optimizer=optimizer, name=name, device=device)




if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args()
    print_args(args=args)
    args.debug = True if args.debug == 1 else False
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run(args=args, current_time=current_time, device=device)
