import os
import sys
import torch
import datetime
import warnings
from Data.read import read_data
from loguru import logger
from rich import print as rprint
from Utils.utils import seed_everything, timeit, read_pickel
from Attacks.config import parse_args
from Attacks.Runs.black_box import run as blackbox
from Attacks.Runs.white_box import run as whitebox
from Attacks.Utils.utils import print_args, init_history, get_name
from hashlib import md5

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
warnings.filterwarnings("ignore")



def run(args, current_time, device):

    data_hist, model_hist, att_hist = init_history(args=args)
    name = get_name(args=args, current_date=current_time)

    # read data 
    with timeit(logger, 'init-data'):
        data_name = f"{md5(name['data'].encode()).hexdigest()}.pkl"
        data_path = args.res_path + data_name
        if (os.path.exists(data_path)) & (args.retrain == 0):
            data_hist = read_pickel(file=data_path)
        train_g, val_g, test_g, _ = read_data(args=args, data_name=args.dataset, history=None)
        train_g = train_g.to(device)
        val_g = val_g.to(device)
        test_g = test_g.to(device)
        rprint(f"Node feature device: {train_g.ndata['feat'].device}, Node label device: {train_g.ndata['feat'].device}, "
               f"Src edges device: {train_g.edges()[0].device}, Dst edges device: {train_g.edges()[1].device}")
        tr_loader, va_loader, te_loader = init_loader(args=args, device=device, train_g=train_g, test_g=test_g,
                                                      val_g=val_g)


    if args.attack_mode == 'blackbox':
        blackbox(args=args, current_time=current_time, device=device)
    elif args.attack_mode == 'whitebox':
        whitebox(args=args, current_time=current_time, device=device)


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
