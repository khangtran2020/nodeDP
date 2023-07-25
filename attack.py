import datetime
import warnings
import sys
from config import parse_args_attack
from Data.read import read_data, init_loader
from Models.init import init_model, init_optimizer
from Runs.run_clean import run as run_clean
from Runs.run_nodedp import run as run_nodedp
from Utils.utils import *
from loguru import logger
from rich import print as rprint

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
warnings.filterwarnings("ignore")


def retrain(args, current_time, device, history):
    train_g, val_g, test_g = read_data(args=args, data_name=args.dataset, history=history)
    train_g = train_g.to(device)
    val_g = val_g.to(device)
    test_g = test_g.to(device)
    rprint(f"Node feature device: {train_g.ndata['feat'].device}, Node label device: {train_g.ndata['feat'].device}, "
           f"Src edges device: {train_g.edges()[0].device}, Dst edges device: {train_g.edges()[1].device}")
    tr_loader, va_loader, te_loader = init_loader(args=args, device=device, train_g=train_g, test_g=test_g,
                                                  val_g=val_g)

    model = init_model(args=args)
    optimizer = init_optimizer(optimizer_name=args.optimizer, model=model, lr=args.lr)
    tar_name = get_name(args=args, current_date=current_time)
    history['name'] = tar_name
    tr_info = (train_g, tr_loader)
    va_info = va_loader
    te_info = (test_g, te_loader)

    if args.tar_clean == 1:
        run_mode = run_clean
    else:
        run_mode = run_nodedp

    tar_model, tar_history = run_mode(args=args, tr_info=tr_info, va_info=va_info, te_info=te_info, model=model,
                                  optimizer=optimizer, name=tar_name, device=device, history=history)
    return tar_model, tar_history


def run(args, current_time, device):
    history = init_history_attack()
    if args.retrain_tar:
        train_g, val_g, test_g = read_data(args=args, data_name=args.dataset, history=history)
        train_g = train_g.to(device)
        val_g = val_g.to(device)
        test_g = test_g.to(device)
        rprint(
            f"Node feature device: {train_g.ndata['feat'].device}, Node label device: {train_g.ndata['feat'].device}, "
            f"Src edges device: {train_g.edges()[0].device}, Dst edges device: {train_g.edges()[1].device}")
        tr_loader, va_loader, te_loader = init_loader(args=args, device=device, train_g=train_g, test_g=test_g,
                                                      val_g=val_g)

        tar_model = init_model(args=args)
        optimizer = init_optimizer(optimizer_name=args.optimizer, model=tar_model, lr=args.lr)
        name = get_name(args=args, current_date=current_time)

    run_dict = {
        'clean': run_clean,
        'nodedp': run_nodedp
    }
    run_mode = run_dict[args.mode]

    run_mode(args=args, tr_info=tr_info, va_info=va_info, te_info=te_info, model=model,
             optimizer=optimizer, name=name, device=device, history=history)


if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args_attack()
    print_args_attack(args=args)
    args.debug = True if args.debug == 1 else False
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device == 'cpu':
        device = torch.device('cpu')
    rprint(f"DEVICE USING: {device}")
    run(args=args, current_time=current_time, device=device)
