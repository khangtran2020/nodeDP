import os
import sys
import torch
import datetime
import warnings
from loguru import logger
from rich import print as rprint
from Utils.utils import seed_everything, timeit, read_pickel
from Attacks.config import parse_args
from Attacks.Runs.black_box import run as blackbox
from Attacks.Runs.white_box import run as whitebox
from Attacks.Runs.wb_simple import run as wanal
from Attacks.Utils.utils import print_args, init_history, get_name, save_dict
from Attacks.Utils.data_utils import shadow_split, shadow_split_whitebox_extreme, shadow_split_whitebox, read_data, shadow_split_whitebox_subgraph, shadow_split_whitebox_drop, shadow_split_whitebox_drop_ratio
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
            rprint(f"History exist, exist_data set to: {exist_data}")
        
        train_g, val_g, test_g, graph = read_data(args=args, history=data_hist, exist=exist_data)
        if args.att_submode == 'blackbox':
            shadow_split(graph=train_g, ratio=args.sha_ratio, history=data_hist, exist=exist_data)
        elif args.att_submode == 'whitebox':
            shadow_graph = shadow_split_whitebox(graph=graph, ratio=args.sha_ratio, history=data_hist, exist=exist_data, diag=True)
        elif args.att_submode == 'wbextreme':
            shadow_graph = shadow_split_whitebox_extreme(graph=graph, ratio=args.sha_ratio, history=data_hist, exist=exist_data, diag=True)
        elif args.att_submode == 'wbsubgraph':
            shadow_graph = shadow_split_whitebox_subgraph(graph=graph, tr_graph=train_g, te_graph=test_g, n_layer=args.n_layers,
                                                          max_nei=args.n_neighbor, ratio=args.sha_ratio, history=data_hist, exist=exist_data, diag=True)
        elif args.att_submode == 'drop':
            shadow_graph = shadow_split_whitebox_drop(graph=graph, ratio=args.sha_ratio, history=data_hist, exist=exist_data, diag=True)
        elif args.att_submode == 'dropdens':
            shadow_graph = shadow_split_whitebox_drop_ratio(graph=graph, ratio=args.sha_ratio, history=data_hist, exist=exist_data, diag=True, density=0.1)
        
        sha_src_edge, sha_dst_edge = shadow_graph.edges()
        sha_nodes = shadow_graph.nodes()

        tr_src_edge, tr_dst_edge = train_g.edges()
        tr_nodes = train_g.nodes()

        te_src_edge, te_dst_edge = test_g.edges()
        te_nodes = test_g.nodes()

        rprint(f"TRAIN graph has: Average degree {train_g.in_degrees().float().mean().item()}, {tr_nodes.size(dim=0)} nodes, {tr_src_edge.size(dim=0)} edges => density {tr_src_edge.size(dim=0) / tr_nodes.size(dim=0) + 1e-12}.")
        rprint(f"TEST graph has: Average degree {test_g.in_degrees().float().mean().item()}, {te_nodes.size(dim=0)} nodes, {te_src_edge.size(dim=0)} edges => density {te_src_edge.size(dim=0) / te_nodes.size(dim=0) + 1e-12}.")
        rprint(f"SHADOW graph has: Average degree {shadow_graph.in_degrees().float().mean().item()}, {sha_nodes.size(dim=0)} nodes, {sha_src_edge.size(dim=0)} edges => density {sha_src_edge.size(dim=0) / sha_nodes.size(dim=0) + 1e-12}.")

        if exist_data == False:
            save_dict(path=data_path, dct=data_hist)

        train_g = train_g.to(device)
        val_g = val_g.to(device)
        test_g = test_g.to(device)
    
    with timeit(logger, 'init-model'):
        model_name = f"{md5(name['model'].encode()).hexdigest()}.pt"
        model_path = args.save_path + model_name
        target_model_name = f"{md5(name['model'].encode()).hexdigest()}.pkl"
        target_model_path = args.res_path + target_model_name

        rprint(f"Model pt exist {os.path.exists(model_path)}, Retraining {bool(args.retrain)}, Target model dict exist: {os.path.exists(target_model_path)}")

        if (os.path.exists(model_path)) & (args.retrain == 0) & (os.path.exists(target_model_path)): 
            exist_model = True
            rprint(f"Model exist, exist_model set to: {exist_model}")
            target_model_name = f"{md5(name['data'].encode()).hexdigest()}.pkl"
            target_model_path = args.res_path + target_model_name
            model_hist = read_pickel(file=target_model_path)

        model = init_model(args=args)
        if exist_model: 
            model.load_state_dict(torch.load(model_path))
            rprint(f"Model exist, loaded previous trained model")

    args.exist_data = exist_data
    args.exist_model = exist_model
    history = (model_hist, att_hist)

    if args.att_mode == 'blackbox':
        model_hist, att_hist = blackbox(args=args, graph=(train_g, val_g, test_g), model=model, device=device, history=history, name=name)
    elif args.att_mode == 'whitebox':
        model_hist, att_hist = whitebox(args=args, graph=(train_g, val_g, test_g, shadow_graph), model=model, device=device, history=history, name=name)
    elif args.att_mode == 'wanal':
        model_hist, att_hist = wanal(args=args, graph=(train_g, val_g, test_g, shadow_graph), model=model, device=device, history=history, name=name)

    general_hist = {
        'data': data_hist,
        'model': model_hist,
        'att': att_hist
    }
    general_path = args.res_path + f"{name['general']}.pkl"
    save_dict(path=general_path, dct=general_hist)
    rprint(f"Saved result at path {general_path}")

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
