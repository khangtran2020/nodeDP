import datetime
import warnings
import torch
import dgl

from config import parse_args
from Data.read import read_data, init_loader
from Utils.utils import seed_everything, get_name
from Models.init import init_model, init_optimizer
from Runs.run_clean import run as run_clean
from Runs.run_dp import run as run_dp

warnings.filterwarnings("ignore")


def run(args, current_time, device):
    train_g, test_g, folds = read_data(args=args, data_name=args.dataset, ratio=args.ratio)

    # create dataloader
    tr_loader, val_loader, te_loader = init_loader(args=args, device=device, train_g=train_g, test_g=test_g,
                                                   num_fold=folds, fold=0)

    print("One batch:", next(iter(tr_loader)))
    return
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
