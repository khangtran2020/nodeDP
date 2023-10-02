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
# from Attacks.Runs.white_box import run as whitebox
from Attacks.Utils.utils import print_args, init_history, get_name, save_dict
from Attacks.Utils.data_utils import shadow_split
from Models.init import init_model
from hashlib import md5

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
warnings.filterwarnings("ignore")



def run(args, current_time, device):

    data_hist, model_hist, att_hist = init_history(args=args)
    name = get_name(args=args, current_date=current_time)
    exist_data = False
    exist_model = False

    # read data 
    with timeit(logger, 'init-data'):
        data_name = f"{md5(name['data'].encode()).hexdigest()}.pkl"
        data_path = args.res_path + data_name
        if (os.path.exists(data_path)) & (args.retrain == 0):
            data_hist = read_pickel(file=data_path)
            exist_data = True

        train_g, val_g, test_g, _ = read_data(args=args, data_name=args.dataset, history=data_hist, exist=exist_data)
        shadow_split(graph=train_g, ratio=args.sha_ratio, history=data_hist, exist=exist_data)

        if exist_data == False:
            save_dict(path=data_path, dct=data_hist)

        train_g = train_g.to(device)
        val_g = val_g.to(device)
        test_g = test_g.to(device)
        rprint(f"Node feature device: {train_g.ndata['feat'].device}, Node label device: {train_g.ndata['feat'].device}, "
               f"Src edges device: {train_g.edges()[0].device}, Dst edges device: {train_g.edges()[1].device}")
    
    with timeit(logger, 'init-model'):
        model_name = f"{md5(name['model'].encode()).hexdigest()}.pt"
        model_path = args.res_path + model_name
        target_model_name = f"{md5(name['model'].encode()).hexdigest()}.pkl"
        target_model_path = args.res_path + target_model_name

        if (os.path.exists(model_path)) & (args.retrain == 0) & (os.path.exists(target_model_path)): 
            exist_model = True
            target_model_name = f"{md5(name['data'].encode()).hexdigest()}.pkl"
            target_model_path = args.res_path + target_model_name
            model_hist = read_pickel(file=target_model_path)
            if (os.path.exists(data_path)) & (args.retrain == 0):
                data_hist = read_pickel(file=data_path)
                exist_data = True

        model = init_model(args=args)
        if exist_model: 
            model.load_state_dict(torch.load(model_path))
        
    args.exist_data = exist_data
    args.exist_model = exist_model
    history = (model_hist, att_hist)

    if args.attack_mode == 'blackbox':
        blackbox(args=args, graph=(train_g, val_g, test_g), model=model, device=device, history=history, name=name)
    # elif args.attack_mode == 'whitebox':
    #     whitebox(args=args,graph=(train_g, val_g, test_g), model=model, device=device, history=history, name=name)


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
