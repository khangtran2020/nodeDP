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
from Runs.run_grad_clean import run as run_grad_clean
from Utils.utils import *
from loguru import logger
from rich import print as rprint

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
warnings.filterwarnings("ignore")


def run(args, current_time, device):

    if args.mode in ['clean', 'grad_clean']:
        history = init_history_clean()
    else:
        history = init_history_nodeDP()
    save_args_to_history(args=args, history=history)

    with timeit(logger, 'init-data'):
        train_g, val_g, test_g, _ = read_data(args=args, data_name=args.dataset, history=history)
        train_g = train_g.to(device)
        val_g = val_g.to(device)
        test_g = test_g.to(device)
        rprint(f"Node feature device: {train_g.ndata['feat'].device}, Node label device: {train_g.ndata['feat'].device}, "
               f"Src edges device: {train_g.edges()[0].device}, Dst edges device: {train_g.edges()[1].device}")
        tr_loader, va_loader, te_loader = init_loader(args=args, device=device, train_g=train_g, test_g=test_g,
                                                      val_g=val_g)

    model = init_model(args=args)
    optimizer = init_optimizer(optimizer_name=args.optimizer, model=model, lr=args.lr)
    name = get_name(args=args, current_date=current_time)
    history['name'] = name
    tr_info = (train_g, tr_loader)
    va_info = va_loader
    te_info = (test_g, te_loader)

    run_dict = {
        'clean': run_clean,
        'nodedp': run_nodedp,
        'mlp': run_mlp,
        'grad': run_grad_inspect,
        'grad_clean': run_grad_clean
    }
    run_mode = run_dict[args.mode]

    _, _ = run_mode(args=args, tr_info=tr_info, va_info=va_info, te_info=te_info, model=model,
                    optimizer=optimizer, name=name, device=device, history=history)


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
