import os
import sys
import torch
import torchmetrics
from Models.init import init_model, init_optimizer
from Runs.run_clean import run as run_clean
from Runs.run_nodedp import run as run_nodedp
from loguru import logger
from rich import print as rprint
from Attacks.Data.read_attack import read_data, graph_split, init_shadow_loader, generate_attack_samples, init_loader
from Attacks.Utils.utils import save_dict, generate_name, read_pickel, init_history, timeit
from Attacks.Model.train_eval import train_shadow, train_attack, eval_attack_step
from Attacks.Data.dataset import Data
from Models.models import NN

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")


def retrain(args, name, train_g, val_g, test_g, model, optimizer, history, device):
    train_g = train_g.to(device)
    val_g = val_g.to(device)
    test_g = test_g.to(device)
    rprint(f"Node feature device: {train_g.ndata['feat'].device}, Node label device: {train_g.ndata['feat'].device}, "
           f"Src edges device: {train_g.edges()[0].device}, Dst edges device: {train_g.edges()[1].device}")
    tr_loader, va_loader, te_loader = init_loader(args=args, device=device, train_g=train_g, test_g=test_g,
                                                  val_g=val_g)
    
    tr_info = (train_g, tr_loader)
    va_info = va_loader
    te_info = (test_g, te_loader)

    if args.mode == 'clean':
        run_mode = run_clean
    else:
        run_mode = run_nodedp

    tar_model, hist_dict = run_mode(args=args, tr_info=tr_info, va_info=va_info, te_info=te_info, model=model,
                            optimizer=optimizer, name=name, device=device, history=history['targeted_model'], mode='attack')
    history['targeted_model'] = hist_dict
    return tar_model

def run(args, current_date, device):

    name = generate_name(args=args)
    history_path = args.res_path + name['history']
    hist_exist = False

    if os.path.exists(history_path):
        history = read_pickel(file=history_path)
        hist_exist = True
    else:
        history = init_history(args=args)

    remain_graph, shadow_graph = read_data(args=args, history=history, hist_exist=hist_exist)
    if hist_exist == False:
        save_dict(path=history_path, dct=history)

    tar_model = init_model(args=args)

    train_g, val_g, test_g = graph_split(graph=remain_graph, drop=True)
    rprint(f"History is exist: {hist_exist}")

    with timeit(logger=logger, task='init-target-model'):

        rprint(f"History is exist: {hist_exist}, are we retraining {args.retrain_tar}")

        if (args.retrain_tar == 0) & (hist_exist == True):
            tar_model_trained_path = args.save_path + name['tar_model_trained']
            if os.path.exists(tar_model_trained_path):
                tar_model.load_state_dict(torch.load(tar_model_trained_path))
                rprint(f"Loaded pretrained model")
            else:
                rprint(f"Pretrained model does not exist, have to re-train")
                args.retrain_tar = 1
        else:
            args.retrain_tar = 1
    
        if args.retrain_tar == 1:
            tar_model_init_path = args.save_path + name['tar_model_init']
            if os.path.exists(tar_model_init_path):
                tar_model.load_state_dict(torch.load(tar_model_init_path))
                rprint(f"Loaded initialized model")
            else:
                torch.save(tar_model.state_dict(), tar_model_init_path)
                rprint(f"Initialized model does not exist, have to re-initialize")
            tar_model_optimizer = init_optimizer(optimizer_name=args.optimizer, model=tar_model, lr=args.lr)
            tar_model = retrain(args=args, name=name['tar_model_trained'], train_g=train_g, val_g=val_g, test_g=test_g, model=tar_model, 
                                optimizer= tar_model_optimizer, history=history, device=device)

    with timeit(logger=logger, task='preparing-shadow-model'):
        
        shadow_graph = shadow_graph.to(device)
        tar_model.to(device)
        with torch.no_grad():
            tar_conf = tar_model.full(shadow_graph, shadow_graph.ndata['feat'])
            shadow_graph.ndata['tar_conf'] = tar_conf

        rprint(f"Shadow confidence: {tar_conf.size()}")
        
        shadow_model = init_model(args=args)
        shadow_optimizer = init_optimizer(optimizer_name=args.optimizer, model=shadow_model, lr=args.lr)
        tr_sh_loader, va_sh_loader = init_shadow_loader(args=args, device=device, graph=shadow_graph)

        shadow_model = train_shadow(args=args, tr_loader=tr_sh_loader, va_loader=va_sh_loader, shadow_model=shadow_model,
                                    epochs=args.sha_epochs, optimizer=shadow_optimizer, name=name['name'], device=device)

    with timeit(logger=logger, task='preparing-attack-data'): 

        with torch.no_grad():
            
            shadow_model.load_state_dict(torch.load(args.save_path + f"{name['name']}_shadow.pt"))
            shadow_model.to(device)
    
            shadow_conf = shadow_model.full(shadow_graph, shadow_graph.ndata['feat'])
            shadow_graph.ndata['shadow_conf'] = shadow_conf

            remain_conf = tar_model.full(remain_graph, remain_graph.ndata['feat'])

            x, y = generate_attack_samples(graph=shadow_graph, conf=shadow_conf, mode='shadow', device=device)
            x_test, y_test = generate_attack_samples(graph=remain_graph, conf=remain_conf, mode='target', device=device)

            x = torch.cat([x, x_test], dim=0)
            y = torch.cat([y, y_test], dim=0)

            num_test = x_test.size(0)
            num_train = int((x.size(0) - num_test) * 0.8)
            num_val = x.size(dim=0) - num_test - num_train

            new_dim = x.size(dim=1)

            tr_data = Data(X=x[:num_train], y=y[:num_train])
            va_data = Data(X=x[num_train:num_train+num_val], y=y[num_train:num_train+num_val])
            te_data = Data(X=x[num_train+num_val:], y=y[num_train+num_val:])
    
    with timeit(logger=logger, task='training-attack-model'):

        tr_loader = torch.utils.data.DataLoader(tr_data, batch_size=args.att_bs, pin_memory=False, drop_last=True, shuffle=True)

        va_loader = torch.utils.data.DataLoader(va_data, batch_size=args.att_bs, num_workers=0, shuffle=False, pin_memory=False, drop_last=False)

        te_loader = torch.utils.data.DataLoader(te_data, batch_size=args.att_bs, num_workers=0, shuffle=False, pin_memory=False, drop_last=False)
        
        attack_model = NN(input_dim=new_dim, hidden_dim=64, output_dim=1, n_layer=3)
        attack_optimizer = init_optimizer(optimizer_name=args.optimizer, model=attack_model, lr=args.att_lr)

        attack_model = train_attack(args=args, tr_loader=tr_loader, va_loader=va_loader, te_loader=te_loader,
                                    attack_model=attack_model, epochs=args.att_epocs, optimizer=attack_optimizer,
                                    name=f"{name['name'] }_attack.pt", device=device)

        attack_model.load_state_dict(torch.load(args.save_path + f"{name['name']}_attack.pt"))
        te_loss, te_auc, topk_auc = eval_attack_step(model=attack_model, device=device, loader=te_loader,
                                        metrics=torchmetrics.classification.BinaryAUROC().to(device),
                                        criterion=torch.nn.BCELoss(), rate=args.topk_rate)
        rprint(f"Attack AUC: {te_auc}, topk AUC {topk_auc}")

        date_str = f'{current_date.day}-{current_date.month}-{current_date.year}_{current_date.hour}-{current_date.minute}'
        save_dict(path=f"{args.res_path}{name['human']}_{date_str}.pkl", dct=history)