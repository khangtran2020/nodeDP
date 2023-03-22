import datetime
import warnings
import torch

from config import parse_args
from Data.read import read_data
from Utils.utils import seed_everything, get_name
from Models.init import init_model, init_optimizer


warnings.filterwarnings("ignore")


def run(args, current_time, device):
    ratio = [1 - args.ratio, args.ratio/2, args.ratio/2]
    train_g, val_g, test_g = read_data(args=args, data_name=args.dataset, ratio=ratio)

    # create dataloader
    if args.mode == 'clean':
        pass
    elif args.mode == 'dp':
        pass

    # init optimizers, models, saving names
    model = init_model(args=args)
    optimizer = init_optimizer(optimizer_name=args.optimizer, model=model, lr=args.lr)
    name = get_name(args=args, current_date=current_time)

    # run


if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run(args=args, current_time=current_time, device=device)