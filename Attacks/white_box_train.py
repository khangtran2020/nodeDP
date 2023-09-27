import sys
import os
import torch
import pickle
import torchmetrics
from Data.read import read_data_attack, read_data, init_loader
from Models.init import init_model, init_optimizer
from Runs.run_clean import run as run_clean
from Runs.run_nodedp import run as run_nodedp
from Attacks.utils import timeit, init_history, get_model_name, get_data_name, read_pickel
from loguru import logger
from rich import print as rprint
from Attacks.train_eval import train_attack, eval_attack_step
from Attacks.helper import generate_attack_samples_white_box, Data
from Models.models import NN
from sklearn.model_selection import train_test_split

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")


def retrain(args, train_g, val_g, test_g, current_time, device, history):
    train_g = train_g.to(device)
    val_g = val_g.to(device)
    test_g = test_g.to(device)
    rprint(f"Node feature device: {train_g.ndata['feat'].device}, Node label device: {train_g.ndata['feat'].device}, "
           f"Src edges device: {train_g.edges()[0].device}, Dst edges device: {train_g.edges()[1].device}")
    tr_loader, va_loader, te_loader = init_loader(args=args, device=device, train_g=train_g, test_g=test_g,
                                                  val_g=val_g)
    
    """
        Initialize model and optimizer
    """

    model_name_init = get_model_name(history=history, mode='target', state='init')
    model_path_init = args.save_path + model_name_init
    model = init_model(args=args)
    if os.path.exists(path=model_path_init):
        model.load_state_dict(torch.load(model_path_init))
        rprint("Loaded previous model initialization")
    else:
        torch.save(model.state_dict(), model_path_init)
        rprint("Saved model initialization")

    optimizer = init_optimizer(optimizer_name=args.optimizer, model=model, lr=args.lr)
    

    tr_info = (train_g, tr_loader)
    va_info = va_loader
    te_info = (test_g, te_loader)

    if args.tar_clean == 1:
        run_mode = run_clean
    else:
        run_mode = run_nodedp

    model_name_trained = get_model_name(history=history, mode='target', state='trained')
    tar_model, _ = run_mode(args=args, tr_info=tr_info, va_info=va_info, te_info=te_info, 
                                      model=model, optimizer=optimizer, name=model_name_trained, 
                                      device=device, history=history, mode='attack')
    rprint("Finished retraining")
    return tar_model


def run_white_box_train(args, current_time, device):

    """
        Loading target model
    """

    history = init_history(args=args)
    data_name = get_data_name(history=history)
    data_path = args.save_path + data_name
    
    model_name_trained = get_model_name(history=history, mode='target', state='trained')
    model_path_trained = args.save_path + model_name_trained


    with timeit(logger=logger, task='init-target-model'):

        if os.path.exists(path=model_path_trained) & os.path.exists(path=data_path):
            data_dict = read_pickel(file=data_path)
            train_g, val_g, test_g, graph = read_data_attack(args=args, data_name=args.dataset, history=data_dict)
            rprint("Loaded data separation")
            tar_model = init_model(args=args)
            tar_model.load_state_dict(torch.load(model_path_trained))
            rprint("Loaded targeted model")

        else:
            train_g, val_g, test_g, graph = read_data(args=args, data_name=args.dataset, history=history)
            data_dict = {}
            data_dict['tr_id'] = history['tr_id']
            data_dict['va_id'] = history['va_id']
            data_dict['te_id'] = history['te_id']
            with open(data_path, 'wb') as f:
                pickle.dump(data_dict, f)
            rprint("Saved data separation")


            tar_model = retrain(args=args, train_g=train_g, val_g=val_g, test_g=test_g,
                                current_time=current_time, history=history, device=device)
        
    with timeit(logger=logger, task='preparing-attack-data'):
        
        tar_model.to(device)
        g = graph.to(device)
        criter = torch.nn.CrossEntropyLoss(reduction='none')
        x_tr, x_va, x_te, y_tr, y_va, y_te = generate_attack_samples_white_box(graph=g, model=tar_model,
                                                                   criter=criter, device=device)
        
        new_dim = x_tr.size(dim=1)

        tr_data = Data(X=x_tr, y=y_tr)
        va_data = Data(X=x_va, y=y_va)
        te_data = Data(X=x_te, y=y_te)

    # device = torch.device('cpu')
    with timeit(logger=logger, task='train-attack-model'):

        tr_loader = torch.utils.data.DataLoader(tr_data, batch_size=args.batch_size,
                                                pin_memory=False, drop_last=True, shuffle=True)

        va_loader = torch.utils.data.DataLoader(va_data, batch_size=args.batch_size, num_workers=0, shuffle=False,
                                                pin_memory=False, drop_last=False)

        te_loader = torch.utils.data.DataLoader(te_data, batch_size=args.batch_size, num_workers=0, shuffle=False,
                                                pin_memory=False, drop_last=False)
        
        attack_model = NN(input_dim=new_dim, hidden_dim=64, output_dim=1, n_layer=3)
        attack_model_name = get_model_name(history=history, mode='attack', state='trained')
        attack_optimizer = init_optimizer(optimizer_name=args.optimizer, model=attack_model, lr=args.lr)

        attack_model = train_attack(args=args, tr_loader=tr_loader, va_loader=va_loader, te_loader=te_loader,
                                    attack_model=attack_model, epochs=args.attack_epochs, optimizer=attack_optimizer,
                                    name=attack_model_name, device=device)

    attack_model.load_state_dict(torch.load(args.save_path + attack_model_name))
    te_loss, te_auc, top_k = eval_attack_step(model=attack_model, device=device, loader=te_loader,
                                       metrics=torchmetrics.classification.BinaryAUROC().to(device),
                                       criterion=torch.nn.BCELoss(), rate=args.topk_rate)
    rprint(f"Attack AUC: {te_auc}, top k AUC: {top_k}")
