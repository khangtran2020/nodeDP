import datetime
import warnings

import dgl
import numpy as np
import torch
import logging
from config import parse_args
from Data.read import read_data, init_loader
from Utils.utils import seed_everything, get_name, timeit
from Models.init import init_model, init_optimizer
from Runs.run_clean import run as run_clean
from Runs.run_dp import run as run_dp
from Trim.base import AppearDict

warnings.filterwarnings("ignore")
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger('exp')
logger.setLevel(logging.INFO)


def run(args, current_time, device):
    with timeit(logger, 'init-data'):
        train_g, test_g, folds = read_data(args=args, data_name=args.dataset, ratio=args.ratio)
        tr_loader, val_loader, te_loader = init_loader(args=args, device=device, train_g=train_g, test_g=test_g,
                                                       num_fold=folds, fold=0)
    if args.debug:
        with timeit(logger, 'test-counter'):
            dst_node, subgraphs = next(iter(tr_loader))
            dst_node = dst_node.tolist()
            appear_dict = AppearDict(roots=dst_node, subgraphs=subgraphs, trimming_rule='random', k=args.clip_node)
        with timeit(logger, 'test-trimmer'):
            appear_dict.trim()
            subgraphs = appear_dict.subgraphs
            appear_dict_ = AppearDict(roots=dst_node, subgraphs=subgraphs)

    # init optimizers, models, saving names
    model = init_model(args=args)
    optimizer = init_optimizer(optimizer_name=args.optimizer, model=model, lr=args.lr)
    name = get_name(args=args, current_date=current_time)

    # run
    if args.mode == 'clean':
        run_clean(dataloaders=(tr_loader, val_loader, te_loader), model=model, optimizer=optimizer, name=name)
    elif args.mode == 'dp':
        run_dp()


if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run(args=args, current_time=current_time, device=device)
